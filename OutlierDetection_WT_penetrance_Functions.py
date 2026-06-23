import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import os
from ppca import PPCA
import pandas as pd
import numpy as np
from scipy import stats, spatial
from sklearn import svm
from sklearn.decomposition import PCA


def ReadOrWriteINFILE(name, path):
    # Check if SQL files are already merged or not
    for root, dirs, files in os.walk(path):
        if name in files:
            f = open(path + name, 'r')
            INFILE_LIST = [x.strip() for x in f.readlines()]
            return INFILE_LIST


def InitDF():
    """ Returns an empty dictionary with the following attributes."""

    # Single cell identifiers
    filename = np.asarray([])
    col = np.asarray([])
    row = np.asarray([])
    orf = np.asarray([])
    gene = np.asarray([])
    timepoint = np.asarray([])

    # Data for single cell
    data = np.asarray([])
    data_scaled = np.asarray([])
    data_pca = np.asarray([])
    mask_WT = np.asarray([])

    is_inlier = np.asarray([])
    dist = np.asarray([])
    penetrance = np.asarray([])
    pvalue = np.asarray([])
    num_cells = np.asarray([])

    model = {'FileName': filename,
             'Column': col,
             'Row': row,
             'ORF': orf,
             'Gene': gene,
             'Timepoint': timepoint,

             'Data': data,
             'DataScaled': data_scaled,
             'DataPCA': data_pca,
             'mask_WT': mask_WT,

             'Is_Inlier': is_inlier,
             'Distance': dist,
             'Penetrance': penetrance,
             'P-value': pvalue,
             'NumCells': num_cells
             }

    return model


def ExtractPlateInformation(input_df, feature_set, plate, wt, screen_name, combine):
    """ Reads input files.

        Args:
            input_df:       CP output of a plate
            feature_set:    Features to be analyzed
            wt:             WT strains
            screen_name:    FileName

        Returns:
            df:       Extracted information
        """

    # Screen information
    df = InitDF()
    df['FileName'] = np.array(input_df['FileName'])
    df['ORF'] = np.array(input_df['ORF'])
    df['Gene'] = np.array(input_df['Gene'])
    df['Column'] = np.array(input_df['Column'], dtype='int32')
    df['Row'] = np.array(input_df['Row'], dtype='int32')
    df['Data'] = np.array(input_df[feature_set], dtype='float64')
    df['Timepoint'] = np.repeat(plate, len(input_df))

    # Data with only WT cells
    if combine:
        df['mask_WT'] = np.array([x in wt for x in df['ORF']])
    else:
        timepoint = int(screen_name[-2:])
        wt_orf = 'YOR202W'
        if timepoint > 12:
            wt_orf = 'WT'
        df['mask_WT'] = df['ORF'] == wt_orf

    return df


def StandardScaler_fit_transform(data_fit, data_transform=np.array([])):
    """ Adaptation of StandardScaler() with ignoring nan values.
        Scales into zero mean and unit variance in each column.

        Args:
            data_fit:       Data to calculate mean and covariances
            data_transform: Data to scale wrt data_fit

        Returns:
            Scaled data
        """

    mu = np.nanmean(data_fit, axis=0)
    std = np.nanstd(data_fit, axis=0)

    if len(data_transform)>0:
        return (data_transform - mu) / std
    else:
        return (data_fit - mu) / std


def ScalePlate(df):
    """ Returns scaled plate. """

    data_WT = df['Data'][df['mask_WT'] == 1]
    print('\nScaling wrt to all WT cells...')
    df['DataScaled'] = StandardScaler_fit_transform(data_WT, df['Data'])

    return df


def AppendData(df, df_to_add, append_list):
    """ Appends two dictionaries with provided attributes with restrictions.

        Args:
            df:             Dictionary to be added
            df_to_add:      Dictionary to add
            append_list:    Attributes to append

        Returns:
            df:             with appended data
        """

    if len(df['Data']) == 0:
        df['Data'] = df_to_add['Data']
        df['DataScaled'] = df_to_add['DataScaled']
        append_list.remove('Data')
        append_list.remove('DataScaled')
    for i in append_list:
        df[i] = np.append(df[i], df_to_add[i], axis=0)

    return df


def ReadAndScalePlate(file_list, wt, features, screen_name, output, combine):
    """ Reads, scales, and adds single plate to the existing combined dictionary.

        Args:
            filename:       Screen folder location of plate
            wt:             WT ORFs used for scaling
            feature_set:    Feature sets to include
            screen_name:    Screen name

        Returns:
            df:             with new plate information appended
        """

    df = InitDF()

    # Read feature file
    f = open(features, 'r')
    feature_set_all = ['FileName', 'Row', 'Column', 'ORF', 'Gene'] + [x.strip() for x in f.readlines()]
    feature_set = feature_set_all[5:]

    # Read plate
    for filename in file_list:
        print('\nReading: %s' % filename)
        input_df = pd.read_csv(filename, header=0)

        # Extract only WT cells
        input_df = input_df[input_df.ORF.isin(wt)]
        input_df = input_df.reset_index(drop=True)

        # Extract features and remove all nan rows
        input_df = input_df[feature_set_all]
        nan_count = np.asarray(input_df[feature_set].isnull().sum(axis=1))
        input_df = input_df.iloc[nan_count != len(feature_set), :]
        input_df = input_df.reset_index(drop=True)

        # Create dataframe and scale
        plate = filename.split('_')[-2]
        plate_df = ExtractPlateInformation(input_df, feature_set, plate, wt, screen_name, combine)
        plate_df = ScalePlate(plate_df)

        append_list = ['FileName', 'Row', 'Column', 'Timepoint', 'ORF', 'Gene', 'mask_WT', 'Data', 'DataScaled']
        df = AppendData(df, plate_df, append_list)

    return df, feature_set


def DoRegularPCA(df, var, output):
    """ Performs regular PCA with no missing values, returns PCA results. """

    print('\nFeature selection using regular PCA...')
    exp_var = []
    num_PCs = 0

    for i in range(df['DataScaled'].shape[1]):
        pca = PCA(n_components=i)
        pca.fit(df['DataScaled'])
        total_var = sum(pca.explained_variance_ratio_)
        exp_var.append(total_var)
        if total_var > var:
            num_PCs = i
            # np.savetxt(output['PCAExplainedVariance'], pca.explained_variance_ratio_, fmt='%0.4f')
            break

    pca = PCA(n_components=num_PCs)
    df['DataPCA'] = pca.fit_transform(df['DataScaled'])
    PCA_loadings = pca.components_

    return df, exp_var, num_PCs, PCA_loadings


def DoProbabilisticPCA(df, var, output):
    """ Performs probabilistic PCA with no missing values, returns PCA results. """

    print('\nFeature selection using probabilistic PCA...')
    exp_var = [0]
    exp_var_ratio = []
    num_PCs = 0
    ppca = PPCA()
    ppca.fit(df['DataScaled'], d=2)
    exp_var.append(ppca.var_exp[0])
    exp_var_ratio.append(ppca.var_exp[0])
    for i in range(2, df['DataScaled'].shape[1]):
        ppca = PPCA()
        ppca.fit(df['DataScaled'], d=i)
        total_var = ppca.var_exp[i-1]
        exp_var.append(total_var)
        exp_var_ratio.append(ppca.var_exp[i-1] - ppca.var_exp[i-2])
        if total_var > var:
            num_PCs = i
            # np.savetxt(output['PCAExplainedVariance'], exp_var_ratio, fmt='%0.4f')
            break

    ppca = PPCA()
    ppca.fit(df['DataScaled'], d=num_PCs)
    df['DataPCA'] = ppca.transform()
    PPCA_loadings = np.transpose(ppca.C)

    return df, exp_var, num_PCs, PPCA_loadings


def SavePCAFeatureCorr(df, num_PCs, feature_set, output):
    """ Calculates correlation between PCs and raw feature data. """

    pca_feat_corr = pd.DataFrame(columns=feature_set)
    PCA_columns = []
    for i in range(num_PCs):
        PCA_columns.append('PC' + str(i + 1))
        corr = []
        for j in range(len(feature_set)):
            data_raw = np.copy(df['Data'][:, j])
            data_pca = np.copy(df['DataPCA'][:, i])
            where_nan = np.isnan(data_raw)
            data_raw = data_raw[~where_nan]
            data_pca = data_pca[~where_nan]
            corr.append(stats.pearsonr(data_raw, data_pca)[0])
        pca_feat_corr.loc[i,] = corr

    pca_feat_corr = pca_feat_corr.set_index([PCA_columns])
    # pca_feat_corr.to_csv(path_or_buf=output['PCAFeatureCorrelations'])


def DoPCA(df, output, var, feature_set):
    """ Performs PCA or probabilistic PCA depending on nan values.

        Args:
            df:             Existing combined dictionary
            output:         Output filenames
            var:            Minimum explained variance required
            feature_set:    Feature sets to include

        Returns:
            df:             with 'DataPCA' values
        """

    # Checks whether there are nan values and chooses PCA method
    check_nan = np.isnan(df['DataScaled']).any()
    if check_nan:
        df, exp_var, num_PCs, PCA_feature_loadings = DoProbabilisticPCA(df, var, output)
    else:
        df, exp_var, num_PCs, PCA_feature_loadings = DoRegularPCA(df, var, output)

    # Plots total explained variance with each added PC
    plt.plot(exp_var)
    plt.xlabel('Number of PCs')
    plt.ylabel('Total % of variance explained')
    plt.title('Number of PCs to be used = %d / %d features' % (num_PCs, df['Data'].shape[1]))
    # plt.savefig(output['PCAExplainedVariancePlot'])

    # Save PCA data
    SavePCAFeatureCorr(df, num_PCs, feature_set, output)

    return df


def OneClassSVMMethod(df, output, out_threshold, feature_set, kernel='rbf', gamma=0):
    """ Outlier Detection with One-Class SVM Method.

        Args:
            df:             Existing combined dictionary
            output:         Output filenames
            out_threshold:  WT threshold on the right tail to decide on outlier boundary
            feature_set:    Feature sets to include

        Returns:
            df:             With in-outlier information
        """

    print('\nOutlier Detection using OneClassSVM...')

    if gamma == 0:
        gamma = 'auto'

    # Create a subset with only WT cells and fit the model
    clf = svm.OneClassSVM(kernel=kernel, gamma=gamma)
    clf.fit(df['DataPCA'][(df['mask_WT'] == 1) & (df['Timepoint'] == 'T00')])
    dist_to_border = clf.decision_function(df['DataPCA']).ravel()

    # Plot data before the threshold
    is_inlier_before = dist_to_border >= 0
    PlotInAndOutliers(df['DataPCA'], is_inlier_before, output['PCABeforeThreshold'],
                      'Outlier Detection results before thresholding')

    # Threshold and plot data
    threshold = stats.scoreatpercentile(dist_to_border[(df['mask_WT'] == 1) & (df['Timepoint'] == 'T00')], out_threshold)
    df['Is_Inlier'] = dist_to_border >= threshold
    PlotInAndOutliers(df['DataPCA'], df['Is_Inlier'], output['PCAAfterThreshold'],
                      'Outlier Detection results after thresholding')

    PlotDistanceHistogram(df['Is_Inlier'], dist_to_border, threshold, 'Distance to border (-outliers, +inliers)',
                          output['OneClassSVMHistogram'])

    return df


def MahalanobisDistanceMethod(df, output, out_threshold, feature_set):
    """ Outlier Detection with Mahalanobis Distance Method.

        Args:
            df:             Existing combined dictionary
            output:         Output filenames
            out_threshold:  WT threshold on the right tail to decide on outlier boundary
            feature_set:    Feature sets to include

        Returns:
            df:             With in-outlier information
        """

    print('\nOutlier Detection using Mahalanobis Distance...')

    # Create a subset with only WT cells and calculate mean and covariance matrix
    mean_WT = np.mean(df['DataPCA'][(df['mask_WT'] == 1) & (df['Timepoint'] == 'T00')], axis=0)
    inverse_cov_WT = np.linalg.inv(np.cov(np.transpose(df['DataPCA'][df['mask_WT'] == 1])))

    def MahalanobisDistance(x):
        return spatial.distance.mahalanobis(x, mean_WT, inverse_cov_WT)

    # Calculate distances and threshold
    MH_dist = np.apply_along_axis(MahalanobisDistance, axis=1, arr=df['DataPCA'])
    threshold = stats.scoreatpercentile(MH_dist[(df['mask_WT'] == 1) & (df['Timepoint'] == 'T00')], 100.0 - out_threshold)
    df['Is_Inlier'] = MH_dist <= threshold
    PlotInAndOutliers(df['DataPCA'], df['Is_Inlier'], output['PCAThreshold'],
                      'Outlier Detection using MH distance')

    PlotDistanceHistogram(df['Is_Inlier'], MH_dist, threshold, 'Mahalanobis Distance',
                          output['MHDistHistogram'])

    return df


def GMMMethod(df, output, feature_set, prob):
    """ Outlier Detection with GMM Method.

        Args:
            df:             Existing combined dictionary
            output:         Output filenames
            feature_set:    Feature sets to include
            prob:           Minimum probability to belong to the outlier Gaussian

        Returns:
            df:             With in-outlier information
        """

    print('\nOutlier Detection using Mixture of Gaussians...')

    # Create a subset with only WT cells and calculate mean and covariance matrix
    mean_WT = np.mean(df['DataPCA'][(df['mask_WT'] == 1) & (df['Timepoint'] == 'T00')], axis=0)
    inverse_cov_WT = np.linalg.inv(np.cov(np.transpose(df['DataPCA'][df['mask_WT'] == 1])))

    def MahalanobisDistance(x):
        return spatial.distance.mahalanobis(x, mean_WT, inverse_cov_WT)

    # Calculate distances and threshold
    MH_dist = np.apply_along_axis(MahalanobisDistance, axis=1, arr=df['DataPCA'])
    df['Distance'] = MH_dist

    # WT parameters
    dist_wt = df['Distance'][(df['mask_WT'] == 1) & (df['Timepoint'] == 'T00')]
    mu_wt = np.median(dist_wt)
    var_wt = np.var(dist_wt)

    # Initialize all cells to inliers
    df['Is_Inlier'] = np.ones(len(df['Gene']))
    genes = list(set(df['Gene']))

    # Find outliers based on posterior probabilities
    for g in genes:
        mask = df['Gene'] == g
        dist_mt = df['Distance'][mask == 1]
        p, mu, PcGivenx = mogEM(dist_mt.reshape(-1, 1).T, mu_wt, var_wt)
        df['Is_Inlier'][mask == 1] = PcGivenx[0] > (1 - prob)

    PlotInAndOutliers(df['DataPCA'], df['Is_Inlier'], output['PCAThreshold'], 'Outlier Detection using GMM')

    return df


def mogEM(x, mu_wt, var_wt, iters=100, offset=2, minVary=0.01):
    """
    Fits a Mixture of K Diagonal Gaussians on x.

    Inputs:
      x: data with one data vector in each column.
      iters: Number of EM iterations.
      minVary: minimum variance of each Gaussian.

    Returns:
      p: probabilities of clusters (or mixing coefficients).
      mu: mean of the clusters, one in each column.
      vary: variances for the cth cluster, one in each column.
      logLikelihood: log-likelihood of data after every iteration.
    """
    N, T = x.shape

    # Initialize the parameters
    PcGivenx = np.zeros((2, N))
    # Mixing coefficients
    p = np.asarray([0.5, 0.5]).reshape(-1, 1)
    # Likelihood
    logLikelihood = np.zeros((iters, 1))
    # Means
    mu = np.asarray([0, 0]).reshape(1, -1)
    mu[0][0] = mu_wt
    mu[0][1] = mu_wt + offset
    # Variance
    vary = np.asarray([minVary, minVary]).reshape(1, -1)
    vary[0][0] = var_wt
    vary[0][1] = var_wt * 4

    # Do iters iterations of EM
    for i in xrange(iters):
        # Do the E step
        ivary = 1 / vary
        logNorm = np.log(p) - 0.5 * N * np.log(2 * np.pi) - \
                  0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1)
        logPcAndx = np.zeros((2, T))
        for k in xrange(2):
            dis = (x - mu[:, k].reshape(-1, 1)) ** 2
            logPcAndx[k, :] = logNorm[k] - 0.5 * np.sum(ivary[:, k].reshape(-1, 1) * dis, axis=0)

        mx = np.max(logPcAndx, axis=0).reshape(1, -1)
        PcAndx = np.exp(logPcAndx - mx)
        Px = np.sum(PcAndx, axis=0).reshape(1, -1)
        PcGivenx = PcAndx / Px
        logLikelihood[i] = np.sum(np.log(Px) + mx)

        # Do the M step
        # update mixing coefficients
        respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
        p = respTot

        # update mean
        respX = np.zeros((N, 2))
        for k in xrange(2):
            respX[:, k] = np.mean(x * PcGivenx[k, :].reshape(1, -1), axis=1)
        mu = respX / respTot.T
        mu[0][0] = mu_wt

        # update variance
        respDist = np.zeros((N, 2))
        for k in xrange(2):
            respDist[:, k] = np.mean((x - mu[:, k].reshape(-1, 1)) ** 2 * PcGivenx[k, :].reshape(1, -1), axis=1)
        vary = respDist / respTot.T
        vary = (vary >= minVary) * vary + (vary < minVary) * minVary
        vary[0][0] = var_wt

    return p, mu, PcGivenx


def PlotInAndOutliers(X, mask, filename, title):
    """ Plots data with in-outlier information using PC1&2. """

    plt.figure(figsize=(10, 12))
    sns.set(font_scale=1)
    X_all = pd.DataFrame({'PC1': X[:, 0], 'PC2': X[:, 1]})
    X_inliers = pd.DataFrame({'PC1': X[mask == 1, 0], 'PC2': X[mask == 1, 1]})
    X_outliers = pd.DataFrame({'PC1': X[mask == 0, 0], 'PC2': X[mask == 0, 1]})
    g = sns.jointplot(x='PC1', y='PC2', data=X_all, marker='.', color='gray', stat_func=None, s=20)
    g.x = X_outliers.PC1
    g.y = X_outliers.PC2
    g.plot_joint(plt.scatter, marker='.', c='gray', s=20, label='outliers')
    g.x = X_inliers.PC1
    g.y = X_inliers.PC2
    g.plot_joint(plt.scatter, marker='.', c='black', s=20, label='inliers')
    fig = plt.gcf()
    plt.legend(bbox_to_anchor=(1, 1.2), loc='upper left', frameon=True)
    plt.title(title, y=1.2)
    # fig.savefig(filename, bbox_inches='tight')
    fig.clf()


def PlotDistanceHistogram(X, dist, out_threshold, xlabel, output):
    """ Plots distances to WT distribution. """

    fig = plt.gcf()
    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1.5)
    percent_mut = sum(X == 0) / float(X.shape[0]) * 100
    plt.hist(dist, bins='auto', color='black')
    plt.xlabel(xlabel)
    plt.ylabel('Number of cells')
    plt.title('Threshold: %.2f\nPercent Mutants: %.2f ' % (out_threshold, percent_mut))
    # plt.savefig(output)
    fig.clf()


def PValueParameters(df):
    """ Returns WT population with outliers to calculate p-value. """

    WT_cells = len(df['mask_WT'][df['mask_WT'] == True])
    WT_cells_outliers = 0
    for i in range(len(df['mask_WT'])):
        if (df['mask_WT'][i] == True) and (df['Is_Inlier'][i] == False):
            WT_cells_outliers += 1

    return WT_cells, WT_cells_outliers


def PrepareOutputFile(df, output):
    """ Prepares the output file with plate, row and column information.
        Calculates penetrance and p-value.

        Args:
            df:                 Existing combined dictionary
            output:             Output filenames

        Returns:
            final_df_output:    Combined outlier detection results
        """

    print('\nCombining data...')

    final_df = pd.DataFrame()
    append_list = ['FileName', 'ORF', 'Gene', 'Column', 'Row', 'Timepoint', 'Is_Inlier']
    for i in append_list:
        final_df[i] = df[i]
    final_df['Row_Col'] = final_df.Row.map(int).map(str) + '_' + final_df.Column.map(int).map(str)

    output_columns = ['ORF', 'Gene', 'Column', 'Row', 'FileName', 'Timepoint', 'Penetrance', 'Num_cells', 'P-value']
    final_df_output = pd.DataFrame(columns=output_columns)
    this_row = 0
    WT_cells, WT_cells_outliers = PValueParameters(df)

    # Regroup this dataframes by well info
    timepoints = final_df['Timepoint'].unique()
    for t in timepoints:
        final_df_time = final_df[final_df['Timepoint'] == t]
        row_col = final_df_time['Row_Col'].unique().tolist()
        for rc in row_col:
            df_rc = final_df_time[final_df_time['Row_Col'] == rc]
            is_inlier_rc = np.asarray(df_rc.Is_Inlier)
            num_cells = df_rc.shape[0]
            num_outliers = sum(is_inlier_rc == 0)
            pene = float(num_outliers) / num_cells * 100
            pval = 1 - stats.hypergeom.cdf(num_outliers, WT_cells, WT_cells_outliers, num_cells)

            # Append them to corresponding variables
            line = []
            append_list = ['ORF', 'Gene', 'Column', 'Row', 'FileName', 'Timepoint']
            for i in append_list:
                line.append(df_rc[i].unique()[0])
            line.append(pene)
            line.append(num_cells)
            line.append(pval)
            final_df_output.loc[this_row,] = line
            this_row += 1

    # Save into a dataframe
    final_df_output = final_df_output.sort_values('Penetrance', ascending=False)
    final_df_output = final_df_output.reset_index(drop=True)
    final_df_output.to_csv(path_or_buf=output['ODresults'], index=False)

    print('\nPreparing plots...')

    return final_df_output


def PlotPenetrancePlot(df, output):
    # Change ORF names
    hue_order = ['WT - column 1', 'WT - column 2']
    for i in range(len(df)):
        orf_ = df.iloc[i, df.columns.get_loc('ORF')]
        if orf_ == 'WT':
            df.iloc[i, df.columns.get_loc('ORF')] = 'WT - column 2'
        elif orf_ == 'YOR202W':
            df.iloc[i, df.columns.get_loc('ORF')] = 'WT - column 1'

    print('\nPlotting combined penetrance...')

    # Plot penetrance and timepoints
    df['Penetrance'] = df['Penetrance'].astype(float)
    df['Timepoint'] = df['Timepoint'].astype(str)
    df['ORF'] = df['ORF'].astype(str)
    df = df.sort_values('Timepoint', ascending=True)

    plt.figure(figsize=(12, 6))
    sns.set(font_scale=1.5)
    sns.set_style('white')
    cg = sns.boxplot(x='Timepoint', y='Penetrance', hue='ORF', hue_order=hue_order, data=df, palette='Set2')
    cg.set_title('Penetrance over 24 hours')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', frameon=True)
    fig = plt.gcf()
    # fig.savefig(output, bbox_inches='tight')
    fig.clf()
