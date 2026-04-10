# Nil Sahin, February 2017
# Combine separate SQL outputs from CellProfiler and maps the gene names
# Adapted and simplied (from Adrian Verster, March 2014) for Dara Lo - Terminal Phenotypes Project
# usage:
# python /home/morphology/mpg3/Nil/TerminalPhenotypes/Lib/OutlierDetection_WT_penetrance.py
# -p /home/morphology/mpg3/Nil/TerminalPhenotypes/Results/HEH2/2015_July_9_Heh2_R2_Results_v2/

from OutlierDetection_WT_penetrance_Functions import *
from optparse import OptionParser

# Parameters to specify
parser = OptionParser()
parser.add_option('-p', '--path', type='str', dest='path',
                  help='Input files to be analyzed')
parser.add_option('-o', '--output-path', dest='output_path', default='',
                  help='Path to store output files')
parser.add_option('-f', '--features', type='str', dest='feature_set_file',
                  default='FeatureSets_TerminalPhenotypes.txt',
                  help='Feature Sets to include')
parser.add_option('-t', '--threshold', type='float', dest='thres',
                  default=10, help='Fraction of outliers in a population')
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
(options, args) = parser.parse_args()

PATH_OUT = options.path
features_file = options.feature_set_file
variance = options.var
distance_method = options.dist
kernel = options.OneClassSVM_kernel
gamma = options.OneClassSVM_gamma
probability = options.probability
outlier_threshold = options.thres
combine_WT = options.combine

wt_strains = ['YOR202W', 'WT']
FILE_OUT = 'CPout_features_plate_list.txt'
INFILE_LIST = ReadOrWriteINFILE(FILE_OUT, PATH_OUT)
screen_name = INFILE_LIST[0].split('/')[-1][:-16]

# Output files
outdir = options.output_path
output_files = {'PCAExplainedVariance':     os.path.join(outdir, screen_name + '_WT_PCA_explained_variance.txt'),
                'PCAExplainedVariancePlot': os.path.join(outdir, screen_name + '_WT_PCA_explained_variance_plot.png'),
                'PCAFeatureLoadings':       os.path.join(outdir, screen_name + '_WT_PCA_feature_loadings.csv'),
                'PCAFeatureCorrelations':   os.path.join(outdir, screen_name + '_WT_PCA_feature_correlations.csv'),

                'ODresultsplot':            os.path.join(outdir, screen_name + '_WT_OD_results_plot.png'),

                'PCABeforeThreshold':       os.path.join(outdir, screen_name + '_WT_PCA_data_before_threshold.png'),
                'PCAAfterThreshold':        os.path.join(outdir, screen_name + '_WT_PCA_data_after_threshold.png'),
                'PCAThreshold':             os.path.join(outdir, screen_name + '_WT_PCA_data.png'),
                'MHDistHistogram':          os.path.join(outdir, screen_name + '_WT_OD_results_MH_distance_histogram.png'),
                'OneClassSVMHistogram':     os.path.join(outdir, screen_name + '_WT_OD_results_OneClassSVM_distance_histogram.png'),
                'OutlierCellData':          os.path.join(outdir, screen_name + '_WT_OD_results_outlier_cells_rawdata.csv'),
                'ODresults':                os.path.join(outdir, screen_name + '_WT_OD_results.csv')
}

if __name__ == '__main__':
    # Data preprocessing
    df, feature_set = ReadAndScalePlate(INFILE_LIST, wt_strains, features_file, screen_name, output_files, combine_WT)

    # Feature selection
    df = DoPCA(df, output_files, variance, feature_set)

    # Outlier detection
    if distance_method.upper() == 'MAHALANOBIS':
        df = MahalanobisDistanceMethod(df, output_files, outlier_threshold, feature_set)
    elif distance_method.upper() == 'ONECLASSSVM':
        df = OneClassSVMMethod(df, output_files, outlier_threshold, feature_set, kernel, gamma)
    elif distance_method.upper() == 'GMM':
        df = GMMMethod(df, output_files, feature_set, probability)

    # Plot penetrance results
    df_OUT = PrepareOutputFile(df, output_files)
    PlotPenetrancePlot(df_OUT, output_files['ODresultsplot'])
