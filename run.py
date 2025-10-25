from src.methods.prefixCDD.prefixcdd import PrefixCDD
from src.methods.gan.gan import AdaptiveDriftDetector, preprocess_log_for_gan
from timeit import default_timer
from datetime import datetime 
import itertools
import json
import yaml
from src.evaluation import calcPrecisionRecall, calF1Score
import hydra
from omegaconf import DictConfig, OmegaConf
import os


from pm4py import read_xes

list_data = ['sudden_trace_noise0_1000_cd.xes']
def load_params_from_yaml(config_path):
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def run_gan(feature_matrix, vocab_size, max_len, name_log, params, ground_truth, orig_cwd):
    keys = params.keys()
    values = params.values()

    hyper_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
    output_file = os.path.join(os.getcwd(), f'output/{name_log}_gan.txt')
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(f"AdaptiveDriftDetector Experiment Log\n")
        f.write(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total experiments to run: {len(hyper_params)}\n")
        f.write("="*50 + "\n\n")
    for i, params in enumerate(hyper_params):
        init_train_size = params['initial_train_size']
        train_data = feature_matrix[:init_train_size]
        monitoring_stream = feature_matrix[init_train_size:]
        detector = AdaptiveDriftDetector(
            vocab_size = vocab_size,
            num_time_features = 1,
            max_seq_length = max_len,
            buffer_capacity = params['buffer_capacity'],
            seed = 42

        )

        detector.train(train_data, epochs = params['epochs'])
        chunk_size = params['chunk_size']
        detected_drift = []

        for j in range(0, len(monitoring_stream), chunk_size):
            chunk = monitoring_stream[j:j+chunk_size]
            if len(chunk == 0):
                break
            scores, drift_pt, threshold = detector.detect_drift(
                chunk,
                window_size = params['window_size'],
                threshold_factor = params['threshold_factor']
            )
            if drift_pt:
                drift_in_chunk = drift_pt[0]
                absolute_location = init_train_size + j + drift_in_chunk
                detected_drift.append(absolute_location)

                new_normal_data = chunk[drift_in_chunk:]
                if len(new_normal_data) >= detector.min_finetune_size:
                    detector.adapt(new_normal_data)
            else:
                detector._update_buffer(chunk)
        f1_score = calF1Score(detected_drift, ground_truth, lag = 200)
        precision, recall = calcPrecisionRecall(detected_drift, ground_truth, lag = 200)

        print(f"Experiment {i+1} completed., Drifts are detected at {detected_drift}")
        with open(output_file, 'a') as f:
            f.write(f"experiment {i+1}\n")
            f.write(f"{json.dumps(params, indent=4)}\n")
            f.write(f"Drifts are detected at {detected_drift}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1-Score: {f1_score}\n")

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    import hydra
    orig_cwd = hydra.utils.get_original_cwd()
    ground_truth_path = os.path.join(orig_cwd, cfg.ground_truth_file)
    with open(ground_truth_path, 'r') as f:
        ground_truth_dict = json.load(f)

    for log_name in cfg.list_data:
        log_path = os.path.join(orig_cwd, "data_eval", log_name)
        log = read_xes(log_path)
        feature_matrix, activity_to_int, max_len, vocab_size = preprocess_log_for_gan(log)
        params = cfg.params
        ground_truth = ground_truth_dict.get(log_name, [])
        run_gan(feature_matrix, vocab_size, max_len, log_name, params, ground_truth,orig_cwd)

if __name__ == "__main__":
    main()
