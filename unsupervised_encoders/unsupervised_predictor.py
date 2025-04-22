"""Unsupervised learning methods including POS, GREEN, CHROME, ICA, LGI, and PBV."""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from tqdm import tqdm
from evaluation.post_process import *
from unsupervised_encoders.methods.CHROME_DEHAAN import *
from unsupervised_encoders.methods.GREEN import *
from unsupervised_encoders.methods.ICA_POH import *
from unsupervised_encoders.methods.LGI import *
from unsupervised_encoders.methods.PBV import *
from unsupervised_encoders.methods.POS_WANG import *
from unsupervised_encoders.methods.OMIT import *
from evaluation.BlandAltmanPy import BlandAltman

def unsupervised_predict(config, data_loader, method_name):
    """Model evaluation on the testing dataset using unsupervised methods."""
    if data_loader["unsupervised"] is None:
        raise ValueError("No data for unsupervised method predicting")
    print("===Unsupervised Method (" + method_name + ") Predicting ===")
    
    pred_hr_all = []
    gt_hr_all = []
    sbar = tqdm(data_loader["unsupervised"], ncols=80)
    
    for batch_idx, test_batch in enumerate(sbar):
        batch_size = test_batch[0].shape[0]
        for idx in range(batch_size):
            # Retrieve input data and ground truth HR (GT HR is assumed to be a scalar)
            data_input = test_batch[0][idx].cpu().numpy()
            label = test_batch[1][idx].cpu().numpy()
            
            # Print shape of the raw input data for debugging
            # print(f"[Batch {batch_idx}, Sample {idx}] Raw data shape: {data_input.shape}")

            # (C, T, H, W) -> (T, H, W, C)
            data_input = np.transpose(data_input, (1, 2, 3, 0))
            # print(f"[Batch {batch_idx}, Sample {idx}] Transposed data shape: {data_input.shape}")

            
            # Reconstruct BVP signal using the selected unsupervised method
            if method_name == "POS":
                BVP = POS_WANG(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "CHROM":
                BVP = CHROME_DEHAAN(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "ICA":
                BVP = ICA_POH(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "GREEN":
                BVP = GREEN(data_input)
            elif method_name == "LGI":
                BVP = LGI(data_input)
            elif method_name == "PBV":
                BVP = PBV(data_input)
            elif method_name == "OMIT":
                BVP = OMIT(data_input)
            else:
                raise ValueError("Unsupervised method name is wrong!")
            
            # print(f"[Batch {batch_idx}, Sample {idx}] Extracted BVP shape: {BVP.shape}")

            # Print length of extracted BVP signal for debugging
            # print(f"[Batch {batch_idx}, Sample {idx}] Extracted BVP signal length: {len(BVP)}")
            
            # Check if the recovered BVP signal is sufficiently long.
            if len(BVP) < 9:  # Skip
                # print(f"[Batch {batch_idx}, Sample {idx}] Warning: BVP signal length ({len(BVP)}) is too short. Skipping sample.")
                
                # RAW BVP 
                # print(f"[Batch {batch_idx}, Sample {idx}] Raw BVP signal: {BVP}")

                continue

            # Calculate predicted HR using the get_hr function.
            pred_hr = get_hr_from_rppg(BVP, sr=config.UNSUPERVISED.DATA.FS)
            if np.isnan(pred_hr):
                # print("Warning: Predicted HR is NaN. Skipping sample.")
                continue
            
            pred_hr_all.append(pred_hr)
            gt_hr_all.append(label)  # GT HR from the loader
            
    pred_hr_all = np.array(pred_hr_all)
    gt_hr_all = np.array(gt_hr_all)
    
    # Compute evaluation metrics (MAE and RMSE)
    mae = np.mean(np.abs(pred_hr_all - gt_hr_all))
    rmse = np.sqrt(np.mean((pred_hr_all - gt_hr_all) ** 2))
    
    print("Unsupervised method:", method_name)
    print("Predicted HR samples:", pred_hr_all.shape[0])
    print("MAE:", mae)
    print("RMSE:", rmse)


# Standalone function to calculate HR from rPPG using peak detection.
def bandpass_filter(signal, fs=30, lowcut=0.5, highcut=4, order=4):
    """Apply a Butterworth bandpass filter to the input signal."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

def get_hr_from_rppg(rppg_signal, sr=30, min_hr=30, max_hr=180):
    """Extract HR from rPPG signal using peak detection."""
    # Remove DC component.
    rppg_signal = rppg_signal - np.mean(rppg_signal)
    # Bandpass filter the signal.
    filtered_signal = bandpass_filter(rppg_signal, fs=sr, lowcut=0.5, highcut=4, order=4)
    # Detect peaks.
    peaks, _ = find_peaks(filtered_signal, distance=sr*0.6,
                            height=np.mean(filtered_signal) + np.std(filtered_signal))
    if len(peaks) < 2:
        return np.nan
    # Calculate RR intervals and HR values.
    rr_intervals = np.diff(peaks) / sr
    hr_values = 60 / rr_intervals
    # Filter HR values outside the acceptable range.
    hr_values = hr_values[(hr_values >= min_hr) & (hr_values <= max_hr)]
    if len(hr_values) == 0:
        return np.nan
    return np.mean(hr_values)



'''
import numpy as np
from evaluation.post_process import *
from unsupervised_methods.methods.CHROME_DEHAAN import *
from unsupervised_methods.methods.GREEN import *
from unsupervised_methods.methods.ICA_POH import *
from unsupervised_methods.methods.LGI import *
from unsupervised_methods.methods.PBV import *
from unsupervised_methods.methods.POS_WANG import *
from unsupervised_methods.methods.OMIT import *
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman

def unsupervised_predict(config, data_loader, method_name):
    """ Model evaluation on the testing dataset."""
    if data_loader["unsupervised"] is None:
        raise ValueError("No data for unsupervised method predicting")
    print("===Unsupervised Method ( " + method_name + " ) Predicting ===")
    predict_hr_peak_all = []
    gt_hr_peak_all = []
    predict_hr_fft_all = []
    gt_hr_fft_all = []
    SNR_all = []
    MACC_all = []
    sbar = tqdm(data_loader["unsupervised"], ncols=80)
    for _, test_batch in enumerate(sbar):
        batch_size = test_batch[0].shape[0]
        for idx in range(batch_size):
            data_input, labels_input = test_batch[0][idx].cpu().numpy(), test_batch[1][idx].cpu().numpy()
            data_input = data_input[..., :3]
            if method_name == "POS":
                BVP = POS_WANG(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "CHROM":
                BVP = CHROME_DEHAAN(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "ICA":
                BVP = ICA_POH(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "GREEN":
                BVP = GREEN(data_input)
            elif method_name == "LGI":
                BVP = LGI(data_input)
            elif method_name == "PBV":
                BVP = PBV(data_input)
            elif method_name == "OMIT":
                BVP = OMIT(data_input)
            else:
                raise ValueError("unsupervised method name wrong!")

            video_frame_size = test_batch[0].shape[1]
            if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
                window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.UNSUPERVISED.DATA.FS
                if window_frame_size > video_frame_size:
                    window_frame_size = video_frame_size
            else:
                window_frame_size = video_frame_size

            for i in range(0, len(BVP), window_frame_size):
                BVP_window = BVP[i:i+window_frame_size]
                label_window = labels_input[i:i+window_frame_size]

                if len(BVP_window) < 9:
                    print(f"Window frame size of {len(BVP_window)} is smaller than minimum pad length of 9. Window ignored!")
                    continue

                if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                    gt_hr, pre_hr, SNR, macc = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                    fs=config.UNSUPERVISED.DATA.FS, hr_method='Peak')
                    gt_hr_peak_all.append(gt_hr)
                    predict_hr_peak_all.append(pre_hr)
                    SNR_all.append(SNR)
                    MACC_all.append(macc)
                elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                    gt_fft_hr, pre_fft_hr, SNR, macc = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                    fs=config.UNSUPERVISED.DATA.FS, hr_method='FFT')
                    gt_hr_fft_all.append(gt_fft_hr)
                    predict_hr_fft_all.append(pre_fft_hr)
                    SNR_all.append(SNR)
                    MACC_all.append(macc)
                else:
                    raise ValueError("Inference evaluation method name wrong!")
    print("Used Unsupervised Method: " + method_name)

    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'unsupervised_method':
        filename_id = method_name + "_" + config.UNSUPERVISED.DATA.DATASET
    else:
        raise ValueError('unsupervised_predictor.py evaluation only supports unsupervised_method!')

    if config.INFERENCE.EVALUATION_METHOD == "peak detection":
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_peak_all)
        for metric in config.UNSUPERVISED.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                standard_error = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, standard_error))
            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                standard_error = np.std(np.square(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                standard_error = np.std(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(num_test_samples) * 100
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                correlation_coefficient = Pearson_PEAK[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_FFT = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("MACC (avg): {0} +/- {1}".format(MACC_avg, standard_error))
            elif "BA" in metric:
                compare = BlandAltman(gt_hr_peak_all, predict_hr_peak_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                    y_label='Average of rPPG HR and GT PPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    elif config.INFERENCE.EVALUATION_METHOD == "FFT":
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_fft_all)
        for metric in config.UNSUPERVISED.METRICS:
            if metric == "MAE":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
            elif metric == "RMSE":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                standard_error = np.std(np.square(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
            elif metric == "MAPE":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                correlation_coefficient = Pearson_FFT[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_PEAK = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_PEAK, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("MACC (avg): {0} +/- {1}".format(MACC_avg, standard_error))
            elif "BA" in metric:
                compare = BlandAltman(gt_hr_fft_all, predict_hr_fft_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]', 
                    y_label='Average of rPPG HR and GT PPG HR [bpm]', 
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")
'''