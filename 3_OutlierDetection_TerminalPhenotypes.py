# Outlier Detection Method to quantify penetrance gene defects
# Example Usage:
# python OutlierDetection_TerminalPhenotypes.py -i <input_file_T00_rawdata.csv> -o <output_directory>

from OutlierDetection_TerminalPhenotypes_Functions import *
from optparse import OptionParser

# Parameters to specify
parser = OptionParser()
parser.add_option('-i', '--input-file', type='str', dest='input_file',
                  help='Input file to be analyzed')
parser.add_option('-o', '--output-path', dest='output_path', default='',
                  help='Path to store output files')
parser.add_option('-f', '--features', type='str', dest='feature_set_file',
                  default='FeatureSets_TerminalPhenotypes.txt',
                  help='Feature Sets to include')
parser.add_option('-t', '--threshold', type='float', dest='thres',
                  default=10, help='Fraction of outliers in a population'),
parser.add_option('-w', '--combine-wt', action='store_false', dest='combine',
                  default=True, help='Combine wt columns together or not')
parser.add_option('-v', '--variance', type='float', dest='var',
                  default=0.80, help='Variance explained by PCA')
parser.add_option('-d', '--dist-method', type='str', dest='dist',
                  default='OneClassSVM', help='Distance method: Mahalanobis - OneClassSVM - GMM')
parser.add_option('-k', '--OneClassSVM-kernel', type='str', dest='OneClassSVM_kernel',
                  default='rbf', help='Kernel method for OneClassSVM: linear, poly, rbf, sigmoid, precomputed')
parser.add_option('-g', '--OneClassSVM-gamma', type='float', dest='OneClassSVM_gamma',
                  default=0, help='Kernel coefficient for poly, rbf, sigmoid. If 0, 1/n_features will be used.')
parser.add_option('-q' , '--prob', type='float', dest='probability',
                  default=0.9, help='Min probability of a cell belonging to outlier Gaussian from GMM method.')
parser.add_option('-c' , '--cell-threshold', type='float', dest='cell_thresh',
                  default=0, help='Min number of cells to use for penetrance plots.')
(options, args) = parser.parse_args()

filename = options.input_file
features_file = options.feature_set_file
variance = options.var
distance_method = options.dist
kernel = options.OneClassSVM_kernel
gamma = options.OneClassSVM_gamma
probability = options.probability
outlier_threshold = options.thres
combine_WT = options.combine
cell_threshold = options.cell_thresh
path_out = options.output_path

screen_name = filename.split('/')[-1][:-12]
path = filename.replace(screen_name + '_rawdata.csv', '')
wt_strains = ['YOR202W', 'WT']

# Output files
output_files = {'PCAExplainedVariance':     os.path.join(path_out, screen_name + '_PCA_explained_variance.txt'),
                'PCAExplainedVariancePlot': os.path.join(path_out, screen_name + '_PCA_explained_variance_plot.png'),
                'PCAAppliedData':           os.path.join(path_out, screen_name + '_PCA_applied_data.csv'),
                'PCAFeatureLoadings':       os.path.join(path_out, screen_name + '_PCA_feature_loadings.csv'),
                'PCAFeatureCorrelations':   os.path.join(path_out, screen_name + '_PCA_feature_correlations.csv'),
                'PCABeforeThreshold':       os.path.join(path_out, screen_name + '_PCA_data_before_threshold.png'),
                'PCAAfterThreshold':        os.path.join(path_out, screen_name + '_PCA_data_after_threshold.png'),
                'PCAThreshold':             os.path.join(path_out, screen_name + '_PCA_data.png'),
                'PCAPositiveControls':      os.path.join(path_out, screen_name + '_PCA_positive_controls.png'),
                'MHDistHistogram':          os.path.join(path_out, screen_name + '_OD_results_MH_distance_histogram.png'),
                'OneClassSVMHistogram':     os.path.join(path_out, screen_name + '_OD_results_OneClassSVM_distance_histogram.png'),
                'OutlierCellData':          os.path.join(path_out, screen_name + '_OD_results_outlier_cells_rawdata.csv'),
                'ODresults':                os.path.join(path_out, screen_name + '_OD_results.csv'),
                'PenetrancePlate':          os.path.join(path_out, screen_name + '_OD_results_penetrance_heatmap.png'),
                'PenetranceRows':           os.path.join(path_out, screen_name + '_OD_results_penetrance_across_rows.png'),
                'PenetranceCols':           os.path.join(path_out, screen_name + '_OD_results_penetrance_across_columns.png'),
                'PenetranceSorted':         os.path.join(path_out, screen_name + '_OD_results_sorted_penetrance.png'),

                'ROCCurve':                 os.path.join(path_out, screen_name + '_OD_results_ROC_curve.png'),
                'PRCurve':                  os.path.join(path_out, screen_name + '_OD_results_PR_curve.png'),
                'CurveNumbers':             os.path.join(path_out, screen_name + '_OD_results_ROC_PR_curve_numbers.csv'),
                'PenetranceBins':           os.path.join(path_out, screen_name + '_OD_results_penetrance_bins.csv'),
                'ConfusionMatrix':          os.path.join(path_out, screen_name + '_OD_results_confusion_matrix.png'),

                'PenetranceSortedDL':       os.path.join(path_out, screen_name + '_OD_results_sorted_penetrance_DL.png'),
                'ROCCurveDL':               os.path.join(path_out, screen_name + '_OD_results_ROC_curve_DL.png'),
                'PRCurveDL':                os.path.join(path_out, screen_name + '_OD_results_PR_curve_DL.png'),
                'CurveNumbersDL':           os.path.join(path_out, screen_name + '_OD_results_ROC_PR_curve_numbers_DL.csv'),
                'PenetranceBinsDL':         os.path.join(path_out, screen_name + '_OD_results_penetrance_bins_DL.csv'),
                'PenetranceBinsPlotDL':     os.path.join(path_out, screen_name + '_OD_results_penetrance_bins_plot_DL.png'),
                'ConfusionMatrixDL':        os.path.join(path_out, screen_name + '_OD_results_confusion_matrix_DL.png'),

                'PenetranceSortedHF':       os.path.join(path_out, screen_name + '_OD_results_sorted_penetrance_HF.png'),
                'ROCCurveHF':               os.path.join(path_out, screen_name + '_OD_results_ROC_curve_HF.png'),
                'PRCurveHF':                os.path.join(path_out, screen_name + '_OD_results_PR_curve_HF.png'),
                'CurveNumbersHF':           os.path.join(path_out, screen_name + '_OD_results_ROC_PR_curve_numbers_HF.csv'),
                'PenetranceBinsHF':         os.path.join(path_out, screen_name + '_OD_results_penetrance_bins_HF.csv'),
                'PenetranceBinsPlotHF':     os.path.join(path_out, screen_name + '_OD_results_penetrance_bins_plot_HF.png'),
                'ConfusionMatrixHF':        os.path.join(path_out, screen_name + '_OD_results_confusion_matrix_HF.png'),

                'ConfusionMatrixDLHF':      os.path.join(path_out, screen_name + '_OD_results_confusion_matrix_DLHF.png'),

                'ScaledData':               os.path.join(path_out, screen_name + '_scaled_data.csv')
}

if __name__ == '__main__':
    # Data preprocessing
    df, feature_set = ReadAndScalePlate(filename, wt_strains, features_file, screen_name, output_files, combine_WT)

    # Feature selection
    df = DoPCA(df, output_files, variance, feature_set)

    # Outlier detection
    if distance_method.upper() == 'MAHALANOBIS':
        df = MahalanobisDistanceMethod(df, output_files, outlier_threshold, feature_set)
    elif distance_method.upper() == 'ONECLASSSVM':
        df = OneClassSVMMethod(df, output_files, outlier_threshold, feature_set, kernel, gamma)
    elif distance_method.upper() == 'GMM':
        df = GMMMethod(df, output_files, feature_set, probability)

    df_OUT = PrepareOutputFile(df, output_files)
    PlotHeatmaps(df_OUT, output_files)
    PlotPenetrance(df_OUT, 'Row', output_files['PenetranceRows'])
    PlotPenetrance(df_OUT, 'Column', output_files['PenetranceCols'])
    PlotPerformance(screen_name, df_OUT, wt_strains, cell_threshold, output_files)

