# Imports
import numpy as np
import pandas as pd
import torch

# Importing from
    # Internal Dependencies
from lazyqml.Utils.Utils import *
from lazyqml.Factories.Models.fModels import *
from lazyqml.Factories.Preprocessing.fPreprocessing import *
from lazyqml.Utils import get_simulation_type, printer
    # External Libraries
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from multiprocessing import Queue, Process, Pool, Manager
from statistics import mean
from collections import defaultdict
from time import time, sleep

class Dispatcher:
    def __init__(self, nqubits, randomstate, predictions, shots, numPredictors, numLayers, classifiers, ansatzs, backend, embeddings, features, learningRate, epochs, runs, batch, maxSamples, customMetric, customImputerNum, customImputerCat, sequential=False, threshold=22, time=True, cores=-1, custom_circuits=None):
        self.sequential = sequential
        self.threshold = threshold
        self.timeM = time
        self.cores = cores

        self.nqubits = nqubits
        self.randomstate = randomstate
        self.shots = shots
        self.numPredictors = numPredictors
        self.numLayers = numLayers
        self.classifiers = classifiers
        self.ansatzs = ansatzs
        self.backend = backend
        self.embeddings = embeddings
        self.features = features
        self.learningRate = learningRate
        self.epochs = epochs
        self.batch = batch
        self.maxSamples = maxSamples
        self.customMetric = customMetric
        self.customImputerNum = customImputerNum
        self.customImputerCat = customImputerCat
        self.predictions = predictions

        self.custom_circuits = custom_circuits if custom_circuits else None


    def execute_model(self, model_factory_params, X_train, y_train, X_test, y_test, predictions,  customMetric):
        model = ModelFactory().getModel(**model_factory_params)
        preds = []
        accuracy, b_accuracy, f1, custom = 0, 0, 0, 0
        custom = None

        start = time()

        model.fit(X=X_train, y=y_train)
        y_pred = model.predict(X=X_test)

        accuracy += accuracy_score(y_test, y_pred, normalize=True)
        b_accuracy += balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        if customMetric is not None:
            custom = customMetric(y_test, y_pred)

        exeT = time() - start

        return model_factory_params['Nqubits'], model_factory_params['model'], model_factory_params['Embedding'], model_factory_params['Ansatz'], model_factory_params['Max_features'], exeT, accuracy, b_accuracy, f1, custom, preds

    def process_gpu_task(self, queue, results):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        while not queue.empty():
            try:
                item = queue.get_nowait()

                results.append(self.execute_model(*item[1]))  # Store results if needed
                print(self.execute_model(*item[1]))
            except queue.Empty:
                break

    def process_cpu_task(self,cpu_queue, gpu_queue, results):
        torch.set_num_interop_threads(1)
        torch.set_num_threads(1)
        numProcs = psutil.cpu_count(logical=False)
        total_memory = calculate_free_memory()
        available_memory = total_memory
        if not self.sequential:
            if self.cores == -1:
                available_cores = numProcs
            else:
                available_cores = self.cores
        else:
            available_cores = 1

        # Lock para el acceso seguro a los recursos compartidos
        manager = Manager()
        resource_lock = manager.Lock()

        while not cpu_queue.empty():
            try:
                # Determinar número de cores a usar basado en el estado de gpu_queue
                if gpu_queue.empty():
                    max_cores = numProcs
                else:
                    max_cores = max(1, numProcs - 1)

                current_batch = []
                current_cores = 0

                # Recolectar items para procesar mientras haya recursos disponibles
                while current_cores < max_cores and not cpu_queue.empty():
                    try:
                        item = cpu_queue.get_nowait()
                        # printer.print(f"ITEM CPU: {item[0]}")
                        mem_model = item[0][-1]

                        # Verificar si hay recursos suficientes
                        with resource_lock:
                            if available_memory >= mem_model and available_cores >= 1:
                                # printer.print(f"Available Resources - Memory: {available_memory}, Cores: {available_cores}")
                                available_memory -= mem_model
                                available_cores -= 1
                                current_batch.append(item)
                                current_cores += 1
                            else:
                                # printer.print(f"Unavailable Resources - Requirements: {mem_model}, Available: {available_memory}")
                                cpu_queue.put(item)
                                break

                    except Queue.Empty:
                        break

                # Procesar el batch actual si no está vacío
                if current_batch:
                    # printer.print(f"Executing Batch of {len(current_batch)} Jobs")
                    with Pool(processes=len(current_batch)) as pool:
                        # Usamos map de forma síncrona para asegurar que todos los items se procesen
                        batch_results = pool.starmap(self.execute_model, [params[1] for params in current_batch])

                        # Filtramos los resultados None (errores) y los añadimos a results
                        # valid_results = [r for r in batch_results if r is not None]
                        results.extend(batch_results)

                    # Liberar recursos después del procesamiento
                    with resource_lock:
                        # printer.print("Freeing Up Resources")
                        for item in current_batch:
                            mem_model = item[0][-1]
                            available_memory += mem_model
                            available_cores += 1
                            # printer.print(f"Freed - Memory: {available_memory}MB, Cores: {available_cores}")

                # printer.print("Waiting for next batch")
                sleep(0.1)

            except Exception as e:
                printer.print(f"Error in the batch: {str(e)}")
                import traceback
                traceback.print_exc()
                break

    def prepare_custom_models():
        pass

    def dispatch(self, X, y, showTable, folds=10, repeats=5, mode="cross-validation", testsize=0.4):
        """
        ################################################################################
        Preparing Data Structures & Initializing Variables
        ################################################################################
        """
        # Replace the list-based queues with multiprocessing queues
        manager = Manager()
        gpu_queue = Queue()
        cpu_queue = Queue()
        results = manager.list()  # Shared list for results if needed
        # Also keep track of items for printing
        cpu_items = []
        gpu_items = []

        tensor_sim = get_simulation_type() == "tensor"

        RAM = calculate_free_memory()
        VRAM = calculate_free_video_memory()

        """
        ################################################################################
        Generate CV indices once
        ################################################################################
        """
        cv_indices = generate_cv_indices(
            X, y,
            mode=mode,
            n_splits=folds,
            n_repeats=repeats,
            random_state=self.randomstate,
            test_size=testsize
        )


        """
        ################################################################################
        Generating Combinations
        ################################################################################
        """

        t_pre = time()
        combinations = create_combinations(qubits=self.nqubits,
                                        classifiers=self.classifiers,
                                        embeddings=self.embeddings,
                                        features=self.features,
                                        ansatzs=self.ansatzs,
                                        RepeatID=range(repeats),
                                        FoldID=range(folds))
        cancelledQubits = set()
        to_remove = []

        # TODO: Preparar combinaciones custom
        custom_combinations = []
        for c in self.custom_circuits:
            # qubits, name, repeat, fold, circuit, model_params, memModel
            temp_custom_combinations = list(product(list(self.nqubits), [c['name']], range(repeats), range(folds), [c['circuit']], [c['circ_params']]))

            for t in temp_custom_combinations:
                custom_combinations.append(t + (calculate_quantum_memory(t[0]),))

        printer.print(str(custom_combinations))

        for _, combination in enumerate(combinations):
            modelMem = combination[-1]
            if modelMem > RAM and modelMem > VRAM:
                to_remove.append(combination)

        for combination in to_remove:
            combinations.remove(combination)
            cancelledQubits.add(combination[0])

        for val in cancelledQubits:
            printer.print(f"Execution with {val} Qubits are cancelled due to memory constrains -> Memory Required: {calculate_quantum_memory(val)/1024:.2f}GB Out of {calculate_free_memory()/1024:.2f}GB")

        X = pd.DataFrame(X)

        # Prepare all model executions
        for combination in combinations:
            qubits, name, embedding, ansatz, feature, repeat, fold, memModel = combination
            feature = feature if feature is not None else "~"

            # Get indices for this repeat/fold combination
            train_idx, test_idx = get_train_test_split(cv_indices, repeat, fold)

            numClasses = len(np.unique(y))
            adjustedQubits = qubits  # or use adjustQubits if needed
            prepFactory = PreprocessingFactory(adjustedQubits)

            # Process data for this specific combination using pre-generated indices
            X_train_processed, X_test_processed, y_train_processed, y_test_processed = dataProcessing(
                X,
                y,
                prepFactory,
                self.customImputerCat,
                self.customImputerNum,
                train_idx,
                test_idx,
                ansatz=ansatz,
                embedding=embedding
            )

            model_factory_params = {
                "Nqubits": adjustedQubits,
                "model": name,
                "Embedding": embedding,
                "Ansatz": ansatz,
                "N_class": numClasses,
                "backend": self.backend,
                "Shots": self.shots,
                "seed": self.randomstate*repeat,
                "Layers": self.numLayers,
                "Max_samples": self.maxSamples,
                "Max_features": feature,
                "LearningRate": self.learningRate,
                "BatchSize": self.batch,
                "Epoch": self.epochs,
                "numPredictors": self.numPredictors
            }

            # When adding items to queues
            if name == Model.QNN and qubits >= self.threshold and VRAM > memModel:
                model_factory_params["backend"] = Backend.lightningGPU if not tensor_sim else Backend.lightningTensor
                gpu_queue.put((combination,(model_factory_params, X_train_processed, y_train_processed, X_test_processed, y_test_processed, self.predictions, self.customMetric)))
                gpu_items.append(combination)
            else:
                model_factory_params["backend"] = Backend.lightningQubit if not tensor_sim else Backend.defaultTensor
                cpu_queue.put((combination,(model_factory_params, X_train_processed, y_train_processed, X_test_processed, y_test_processed, self.predictions, self.customMetric)))
                cpu_items.append(combination)

        # TODO: Preparar ejecuciones de los modelos custom y añadir a las colas
        for combination in custom_combinations:
            qubits, name, repeat, fold, circuit, model_params, memModel = combination
            feature = feature if feature is not None else "~"

            # Get indices for this repeat/fold combination
            train_idx, test_idx = get_train_test_split(cv_indices, repeat, fold)

            numClasses = len(np.unique(y))
            adjustedQubits = qubits  # or use adjustQubits if needed
            prepFactory = PreprocessingFactory(adjustedQubits)

            # Process data for this specific combination using pre-generated indices
            X_train_processed, X_test_processed, y_train_processed, y_test_processed = dataProcessing(
                X,
                y,
                prepFactory,
                self.customImputerCat,
                self.customImputerNum,
                train_idx,
                test_idx
            )

            model_factory_params = {
                "Nqubits": adjustedQubits,
                "Embedding": Embedding.ALL,
                "Ansatz": Ansatzs.ALL,
                "N_class": numClasses,
                "Max_features": 10,
                "model": name,
                "custom_model": {
                    "circuit": circuit,
                    "circ_params": {"nqubits": adjustedQubits, **model_params}
                }
            }

            printer.print(str(model_params))

            # When adding items to queues
            cpu_queue.put((combination,(model_factory_params, X_train_processed, y_train_processed, X_test_processed, y_test_processed, self.predictions, self.customMetric)))
            cpu_items.append(combination)


        if self.timeM:
            printer.print(f"PREPROCESSING TIME: {time()-t_pre}")

        """
        ################################################################################
        Creating processes
        ################################################################################
        """
        sleep(.001)
        executionTime = time()
        gpu_process = None
        # Start GPU process
        if not gpu_queue.empty():
            gpu_process = Process(target=self.process_gpu_task, args=(gpu_queue, results))
            gpu_process.start()

        # Start CPU processes
        if not cpu_queue.empty():
            self.process_cpu_task(cpu_queue, gpu_queue, results)

        # Wait for all processes to complete
        if gpu_process is not None:
            gpu_process.join()

        executionTime = time()-executionTime
        printer.print(f"Execution TIME: {executionTime}")

        """
        ################################################################################
        Processing results
        ################################################################################
        """
        t_res = time()

        grouped_results = defaultdict(list)
        for result in list(results):
            key = result[:5]
            grouped_results[key].append(result)

        # For benchmarking/testing reasons
        # for key, group in grouped_results.items():
        #     for r in group:
        #         print(r[5], r[11], r[12])

        summary = []
        cols = ["Qubits", "Model", "Embedding", "Ansatz", "Features", "Time taken", "Accuracy", "Balanced Accuracy", "F1 Score"]
        for key, group in grouped_results.items():
            time_taken = sum([r[5] for r in group])
            accuracy = mean([r[6] for r in group])
            balanced_accuracy = mean([r[7] for r in group])
            f1_score = mean([r[8] for r in group])
            if self.customMetric:
                if mode == "hold-out":
                    cols = ["Qubits", "Model", "Embedding", "Ansatz", "Features", "Time taken", "Accuracy", "Balanced Accuracy", "F1 Score", "Custom Metric", "Predictions"]
                    summary.append((key[0], key[1], key[2], key[3], key[4], time_taken, accuracy, balanced_accuracy, f1_score, custom_metric, []))
                else:
                    cols = ["Qubits", "Model", "Embedding", "Ansatz", "Features", "Time taken", "Accuracy", "Balanced Accuracy", "F1 Score", "Custom Metric"]
                    custom_metric = mean([r[9] for r in group])
                    summary.append((key[0], key[1], key[2], key[3], key[4], time_taken, accuracy, balanced_accuracy, f1_score, custom_metric))
            else:
                if mode == "hold-out":
                    cols = ["Qubits", "Model", "Embedding", "Ansatz", "Features", "Time taken", "Accuracy", "Balanced Accuracy", "F1 Score", "Custom Metric", "Predictions"]
                    summary.append((key[0], key[1], key[2], key[3], key[4], time_taken, accuracy, balanced_accuracy, f1_score, [], []))
                else:
                    cols = ["Qubits", "Model", "Embedding", "Ansatz", "Features", "Time taken", "Accuracy", "Balanced Accuracy", "F1 Score"]
                    summary.append((key[0], key[1], key[2], key[3], key[4], time_taken, accuracy, balanced_accuracy, f1_score))
        scores = pd.DataFrame(summary, columns=cols)
        scores = scores.sort_values(by="Balanced Accuracy",ascending=False).reset_index(drop=True)
        if showTable:
            print(scores.to_markdown())

        if self.timeM:
            printer.print(f"RESULTS TIME: {time() - t_res}")
        return scores
