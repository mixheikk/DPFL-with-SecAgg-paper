# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower server."""


import concurrent.futures
import numpy as np
from collections import OrderedDict
from jsonargparse import Namespace
import os
import pickle
import timeit
from logging import DEBUG, INFO
import torch
from typing import Dict, List, Optional, Tuple, Union
import wandb

from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
)
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.common import parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy

#from flwr_customised.server.client_manager import ClientManager
#from .history import History
#from utils import load_client_results

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]


class Server:
    """Flower server."""

    def __init__(
            self, 
            *, 
            client_manager: ClientManager, 
            strategy: Optional[Strategy]=None, 
            checkpoint_folder: str=None, 
            random_seeds: Optional[Dict[str,int]] = None,
            general_args: Namespace = None,
            enable_wandb: bool=False, 
            model_save_name: str=None, 
            checkpoint_frequency: int=None, 
            delete_prev_checkpoint:bool=False,
    ) -> None:
        
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = strategy if strategy is not None else FedAvg()
        self.max_workers: Optional[int] = None
        if checkpoint_folder is not None and checkpoint_folder[-1] != '/':
            checkpoint_folder += '/'
        self.checkpoint_folder = checkpoint_folder
        self.enable_wandb = enable_wandb
        self.random_seeds = random_seeds
        self.general_args = general_args
        self.model_save_name = model_save_name
        self.checkpoint_frequency = checkpoint_frequency
        self.delete_prev_checkpoint = delete_prev_checkpoint
        self.last_checkpoints = []

    def set_max_workers(self, max_workers: Optional[int]) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers

    def set_strategy(self, strategy: Strategy) -> None:
        """Replace server strategy."""
        self.strategy = strategy

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    def save_model(self, filename) -> None:
        tmp_model =  self.client_manager().sample(1,1)[0].client_fn('0').get_client_model()
        params_dict = zip(tmp_model.state_dict().keys(), parameters_to_ndarrays(self.parameters) )
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
        tmp_model.load_state_dict(state_dict, strict=True)
        torch.save(tmp_model.state_dict(), f=filename)

    def delete_file(self, filename) -> None:
        try:
            os.remove(filename)
        except OSError:
            print(f"Couldn't remove file {filename}!")

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float], existing_history: History=None) -> History:
        """Run federated averaging for a number of rounds."""

        if existing_history is None:
            history = History()
            round_offset = 0
        else:
            history = existing_history
            # use largest fit round as offset
            if len(history.metrics_distributed_fit['weighted_loss']) > 0:
                round_offset = max([x[0] for x in history.metrics_distributed_fit['weighted_loss']])
            else:
                raise ValueError('Trying to continue from existing history, but given history has no distributed fit metrics!')
            log(INFO, f"Using existing history, continuing training from global round {round_offset+1}")

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        if round_offset == 0:
            # note: this currently evaluates on with centralised data
            log(INFO, "Evaluating initial parameters")
            res = self.strategy.evaluate(0, parameters=self.parameters)
            if res is not None:
                log(
                    INFO,
                    "initial parameters (loss, other metrics): %s, %s",
                    res[0],
                    res[1],
                )
                history.add_loss_centralized(server_round=0, loss=res[0])
                history.add_metrics_centralized(server_round=0, metrics=res[1])
                if self.enable_wandb:
                    wandb.log({"centralised_val/loss": res[0] if res is not None else None,
                               "centralised_val/metrics": res[1] if res is not None else None,
                               "global update": 0},
                               step = 0)
        
        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1 + round_offset, num_rounds + 1):

            # Train model and replace previous global model
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            
            if res_fit is not None:
                
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if fit_metrics:
                    history.add_metrics_distributed_fit(
                        server_round=current_round, metrics=fit_metrics
                    )
                    if self.enable_wandb:
                        wandb.log({"distributed_fit/metrics": fit_metrics,
                                }, step = current_round)
        
            if parameters_prime:
                self.parameters = parameters_prime

            # Evaluate model using strategy implementation (centralised using evaluate_fn)
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )
                if self.enable_wandb:
                    wandb.log({"centralised_val/loss": loss_cen,
                               "centralised_val/metrics": metrics_cen,
                               "global update": current_round
                               }, step = current_round)

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)

            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )
                    if self.enable_wandb:
                        wandb.log({"distributed_val/loss": loss_fed,
                                "distributed_val/metrics": evaluate_metrics_fed,
                                }, step = current_round)
                    
            # save model checkpoint
            if self.checkpoint_frequency is not None and self.checkpoint_folder is not None and self.model_save_name is not None and (current_round+1) % self.checkpoint_frequency == 0:
                tmp = self.checkpoint_folder + self.model_save_name + f'-round{current_round}.pt'
                log(DEBUG, f'Saving model checkpoint to {tmp}')
                self.save_model(filename=tmp)

            if self.delete_prev_checkpoint and current_round > 0:
                tmp = self.model_save_name + f'-round{current_round-1}.pt'
                log(DEBUG, "Removing previous model checkpoint")
                self.delete_file(tmp)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)

        # save final model
        if self.model_save_name is not None and self.checkpoint_folder is not None :
            tmp = self.checkpoint_folder + '/' + self.model_save_name + '-final.pt'
            log(INFO, f'Saving final model to {tmp}')
            self.save_model(filename=tmp)
            tmp = self.checkpoint_folder + '/' + self.model_save_name + '-final-history.pickle'
            log(DEBUG, f'Saving training history to {tmp}')
            with open(tmp, 'wb') as f:
                pickle.dump(history, file=f)

        return history

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""

        parameters_aggregated, metrics_aggregated, results, failures = None, None, None, None

        # Get clients and their respective instructions from strategy
        # NOTE: need to save instructions if need later on, skip saving for now
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results), # list of results, element per sampled client, contains tuple with client handle and FitRes object
            len(failures),)

        # aggregate results:
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)
        parameters_aggregated, metrics_aggregated = aggregated_result
            
        return parameters_aggregated, metrics_aggregated, (results, failures)
    
    def disconnect_all_clients(self, timeout: Optional[float]) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        clients = [all_clients[k] for k in all_clients.keys()]
        instruction = ReconnectIns(seconds=None)
        client_instructions = [(client_proxy, instruction) for client_proxy in clients]
        _ = reconnect_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

    def _get_initial_parameters(self, timeout: Optional[float]) -> Parameters:
        """Get initial parameters from one of the available clients."""

        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout)
        log(INFO, "Received initial parameters from one random client")
        return get_parameters_res.parameters


def reconnect_clients(
    client_instructions: List[Tuple[ClientProxy, ReconnectIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(reconnect_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, DisconnectRes]] = []
    failures: List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
    client: ClientProxy,
    reconnect: ReconnectIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, DisconnectRes]:
    """Instruct client to disconnect and (optionally) reconnect later."""
    disconnect = client.reconnect(
        reconnect,
        timeout=timeout,
    )
    return client, disconnect


def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures
        )
    return results, failures


def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout=timeout)
    return client, fit_res


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, FitRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_evaluate(
            future=future, results=results, failures=failures
        )
    return results, failures


def evaluate_client(
    client: ClientProxy,
    ins: EvaluateIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins, timeout=timeout)
    return client, evaluate_res


def _handle_finished_future_after_evaluate(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, EvaluateRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)
