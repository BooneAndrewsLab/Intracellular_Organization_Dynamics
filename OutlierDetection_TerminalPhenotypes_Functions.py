import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from ppca import PPCA
import pandas as pd
import numpy as np
import itertools
from scipy import stats, spatial
from sklearn import svm, metrics
from sklearn.decomposition import PCA

def InitDF():
    """ Returns an empty dictionary with the following attributes."""

    # Single cell identifiers
    filename = np.asarray([])
    col = np.asarray([])
    row = np.asarray([])
    orf = np.asarray([])
    gene = np.asarray([])

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

    model = {'Filename': filename,
             'Column': col,
             'Row': row,
             'ORF': orf,
             'Gene': gene,

             'Data': data,
             'DataScaled': data_scaled,
             'DataPCA': data_pca,
             'Mask_WT': mask_WT,

             'Is_Inlier': is_inlier,
             'Distance': dist,
             'Penetrance': penetrance,
             'P-value': pvalue,
             'NumCells': num_cells
             }

    return model


def ExtractPlateInformation(input_df, feature_set, wt, screen_name, combine):
    """ Reads input files.

        Args:
            input_df:       CP output of a plate
            feature_set:    Features to be analyzed
            wt:             WT strains
            screen_name:    Filename

        Returns:
            df:       Extracted information
        """

    # Screen information
    df = InitDF()
    df['ORF'] = np.array(input_df['ORF'])
    df['Gene'] = np.array(input_df['Gene'])
    df['Filename'] = np.array(input_df['FileName'])
    df['Column'] = np.array(input_df['Column'], dtype='int32')
    df['Row'] = np.array(input_df['Row'], dtype='int32')
    df['Data'] = np.array(input_df[feature_set], dtype='float64')

    # Data with only WT cells
    if combine:
        df['Mask_WT'] = np.array([x in wt for x in df['ORF']])
    else:
        timepoint = int(screen_name[-2:])
        wt_orf = 'YOR202W'
        if timepoint > 12:
            wt_orf = 'WT'
        df['Mask_WT'] = df['ORF'] == wt_orf

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

    data_WT = df['Data'][df['Mask_WT'] == 1]
    print('\nScaling wrt to all WT cells...')
    df['DataScaled'] = StandardScaler_fit_transform(data_WT, df['Data'])

    return df


def SaveScaledData(df, feature_set, output, screen_name):
    """ Saves scaled data for each plate."""

    data_scaled_columns = ['ORF', 'Gene', 'Filename', 'Column', 'Row'] + feature_set
    data_scaled_output = np.concatenate((df['ORF'].reshape(-1, 1),
                                         df['Gene'].reshape(-1, 1),
                                         df['Filename'].reshape(-1, 1),
                                         df['Column'].reshape(-1, 1),
                                         df['Row'].reshape(-1, 1),
                                         df['DataScaled']), axis=1)
    data_scaled_output_df = pd.DataFrame(data=data_scaled_output, columns=data_scaled_columns)
    data_scaled_output_df = data_scaled_output_df.fillna('')
    data_scaled_output_df.to_csv(path_or_buf = output['ScaledData'], index=False)


def ReadAndScalePlate(filename, wt, features, screen_name, output, combine):
    """ Reads, scales, and adds single plate to the existing combined dictionary.

        Args:
            filename:       Screen folder location of plate
            wt:             WT ORFs used for scaling
            feature_set:    Feature sets to include
            screen_name:    Screen name

        Returns:
            df:             with new plate information appended
        """

    # Read feature file
    f = open(features, 'r')
    feature_set = [x.strip() for x in f.readlines()]
    feature_set = ['FileName', 'Row', 'Column', 'ORF', 'Gene'] + feature_set

    # Read plate
    print('\nReading: %s' % screen_name)
    input_df = pd.read_csv(filename, header=0)

    # Extract features and remove all nan rows
    input_df = input_df[feature_set]
    feature_set = feature_set[5:]
    input_df[feature_set] = input_df[feature_set].replace(to_replace=r'[a-zA-Z]+', value=np.nan, regex=True)
    nan_count = np.asarray(input_df[feature_set].isnull().sum(axis=1))
    input_df = input_df.iloc[nan_count != len(feature_set), :]
    input_df = input_df.reset_index(drop=True)

    # Create dataframe and scale
    df = ExtractPlateInformation(input_df, feature_set, wt, screen_name, combine)
    df = ScalePlate(df)
    SaveScaledData(df, feature_set, output, screen_name)

    return df, feature_set


def SavePCAData(df, num_PCs, feature_loadings, feature_set, output):
    """ Saves PCA applied data and PC coefficients. """

    data_PCA_columns = ['ORF', 'Gene', 'Filename', 'Column', 'Row']
    PCA_columns = []
    for x in range(1, num_PCs + 1):
        PCA_columns.append('PC' + str(x))
    data_PCA_columns += PCA_columns

    data_PCA_output = np.concatenate((df['ORF'].reshape(-1, 1),
                                      df['Gene'].reshape(-1, 1),
                                      df['Filename'].reshape(-1, 1),
                                      df['Column'].reshape(-1, 1),
                                      df['Row'].reshape(-1, 1),
                                      df['DataPCA']), axis=1)

    data_PCA_output_df = pd.DataFrame(data=data_PCA_output, columns=data_PCA_columns)
    data_PCA_output_df.to_csv(path_or_buf=output['PCAAppliedData'], index=False)

    PCA_feature_coef = pd.DataFrame(feature_loadings, index=PCA_columns, columns=feature_set)
    PCA_feature_coef.to_csv(path_or_buf=output['PCAFeatureLoadings'])


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
            np.savetxt(output['PCAExplainedVariance'], pca.explained_variance_ratio_, fmt='%0.4f')
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
    ppca.fit(data=df['DataScaled'], d=2)
    exp_var.append(ppca.var_exp[0])
    exp_var_ratio.append(ppca.var_exp[0])
    for i in range(2, df['DataScaled'].shape[1]):
        ppca = PPCA()
        ppca.fit(data=df['DataScaled'], d=i)
        total_var = ppca.var_exp[i-1]
        exp_var.append(total_var)
        exp_var_ratio.append(ppca.var_exp[i-1] - ppca.var_exp[i-2])
        if total_var > var:
            num_PCs = i
            np.savetxt(output['PCAExplainedVariance'], exp_var_ratio, fmt='%0.4f')
            break

    ppca = PPCA()
    ppca.fit(data=df['DataScaled'], d=num_PCs)
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
    pca_feat_corr.to_csv(path_or_buf=output['PCAFeatureCorrelations'])


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
    fig = plt.gcf()
    fig.savefig(output['PCAExplainedVariancePlot'])
    fig.clf()
    plt.close(fig)

    # Save PCA data
    SavePCAData(df, num_PCs, PCA_feature_loadings, feature_set, output)
    SavePCAFeatureCorr(df, num_PCs, feature_set, output)

    return df


def SaveOutlierData(df, feature_set, output):
    """ Saves scaled data for outlier cells. """

    data_columns = ['ORF', 'Gene', 'Filename', 'Column', 'Row'] + feature_set
    data_output = np.concatenate((df['ORF'][df['Is_Inlier'] == 0].reshape(-1, 1),
                                  df['Gene'][df['Is_Inlier'] == 0].reshape(-1, 1),
                                  df['Filename'][df['Is_Inlier'] == 0].reshape(-1, 1),
                                  df['Column'][df['Is_Inlier'] == 0].reshape(-1, 1),
                                  df['Row'][df['Is_Inlier'] == 0].reshape(-1, 1),
                                  df['DataScaled'][df['Is_Inlier'] == 0]), axis=1)

    data_output_df = pd.DataFrame(data=data_output, columns=data_columns)
    data_output_df = data_output_df.fillna('')
    data_output_df.to_csv(path_or_buf=output['OutlierCellData'], index=False)


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
    clf.fit(df['DataPCA'][df['Mask_WT'] == 1])
    dist_to_border = clf.decision_function(df['DataPCA']).ravel()

    # Plot data before the threshold
    is_inlier_before = dist_to_border >= 0
    PlotInAndOutliers(df['DataPCA'], is_inlier_before, output['PCABeforeThreshold'],
                      'Outlier Detection results before thresholding')

    # Threshold and plot data
    threshold = stats.scoreatpercentile(dist_to_border[df['Mask_WT'] == 1], out_threshold)
    df['Is_Inlier'] = dist_to_border >= threshold
    PlotInAndOutliers(df['DataPCA'], df['Is_Inlier'], output['PCAAfterThreshold'],
                      'Outlier Detection results after thresholding')

    # Save outlier cells
    SaveOutlierData(df, feature_set, output)

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
    mean_WT = np.mean(df['DataPCA'][df['Mask_WT'] == 1], axis=0)
    inverse_cov_WT = np.linalg.inv(np.cov(np.transpose(df['DataPCA'][df['Mask_WT'] == 1])))

    def MahalanobisDistance(x):
        return spatial.distance.mahalanobis(x, mean_WT, inverse_cov_WT)

    # Calculate distances and threshold
    MH_dist = np.apply_along_axis(MahalanobisDistance, axis=1, arr=df['DataPCA'])
    threshold = stats.scoreatpercentile(MH_dist[df['Mask_WT'] == 1], 100.0 - out_threshold)
    df['Is_Inlier'] = MH_dist <= threshold
    PlotInAndOutliers(df['DataPCA'], df['Is_Inlier'], output['PCAThreshold'],
                      'Outlier Detection using MH distance')

    # Save outlier cells
    SaveOutlierData(df, feature_set, output)

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
    mean_WT = np.mean(df['DataPCA'][df['Mask_WT'] == 1], axis=0)
    inverse_cov_WT = np.linalg.inv(np.cov(np.transpose(df['DataPCA'][df['Mask_WT'] == 1])))

    def MahalanobisDistance(x):
        return spatial.distance.mahalanobis(x, mean_WT, inverse_cov_WT)

    # Calculate distances and threshold
    MH_dist = np.apply_along_axis(MahalanobisDistance, axis=1, arr=df['DataPCA'])
    df['Distance'] = MH_dist

    # WT parameters
    dist_wt = df['Distance'][df['Mask_WT'] == 1]
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

    # Save outlier CellIDs
    SaveOutlierData(df, feature_set, output)

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
    fig.savefig(filename, bbox_inches='tight')
    fig.clf()
    plt.close(fig)


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
    plt.savefig(output)
    fig.clf()
    plt.close(fig)


def PValueParameters(df):
    """ Returns WT population with outliers to calculate p-value. """

    WT_cells = len(df['Mask_WT'][df['Mask_WT'] == True])
    WT_cells_outliers = 0
    for i in range(len(df['Mask_WT'])):
        if (df['Mask_WT'][i] == True) and (df['Is_Inlier'][i] == False):
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
    append_list = ['Filename', 'ORF', 'Gene', 'Column', 'Row', 'Is_Inlier']
    for i in append_list:
        final_df[i] = df[i]

    output_columns = ['ORF', 'Gene', 'Column', 'Row', 'Filename', 'Penetrance', 'Num_cells', 'P-value']
    final_df_output = pd.DataFrame(columns=output_columns)
    this_row = 0
    WT_cells, WT_cells_outliers = PValueParameters(df)

    # Regroup this dataframes by well info
    wells = final_df['Filename'].unique().tolist()
    for w in wells:
        df_well = final_df[final_df['Filename'] == w]
        is_inlier_well = np.asarray(df_well.Is_Inlier)

        num_cells = df_well.shape[0]
        num_outliers = sum(is_inlier_well == 0)
        pene = float(num_outliers) / num_cells * 100
        pval = 1 - stats.hypergeom.cdf(num_outliers, WT_cells, WT_cells_outliers, num_cells)

        # Append them to corresponding variables
        line = []
        append_list = ['ORF', 'Gene', 'Column', 'Row', 'Filename']
        for i in append_list:
            line.append(df_well[i].unique()[0])
        line.append(pene)
        line.append(num_cells)
        line.append(pval)
        final_df_output.loc[this_row,] = line
        this_row += 1

    # Save into a dataframe
    final_df_output = final_df_output.sort_values('Penetrance', ascending=False)
    final_df_output = final_df_output.reset_index(drop=True)
    final_df_output.to_csv(path_or_buf=output['ODresults'], index=False)

    print('\nPreparing plots...\n')

    return final_df_output


def PlotHeatmaps(df, output):
    """ Plots penetrance values for each plate's wells. """

    penetrance_df = np.ndarray(shape=(16,24), buffer=np.repeat(np.nan, 384))
    for i in range(len(df)):
        row = int(df.iloc[i, df.columns.get_loc('Row')] - 1)
        column = int(df.iloc[i, df.columns.get_loc('Column')] - 1)
        penetrance_df[row][column] = df.iloc[i, df.columns.get_loc('Penetrance')]
    sns.set(font_scale=1.5)
    sns.set_style()
    plt.figure(figsize=(14, 8))
    mask = penetrance_df == np.nan
    cg = sns.heatmap(penetrance_df, linewidth=.4, mask=mask, vmin=0, vmax=100, cmap='copper_r')
    cg.set_title('Penetrance')
    cg.set_xticklabels(range(1, 25))
    cg.set_yticklabels(list(reversed(range(1,17))))
    fig = plt.gcf()
    fig.savefig(output['PenetrancePlate'], bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def PlotPenetrance(df, orient, output):
    """ Plots penetrance values for each plate's rows and column with a violin plot. """

    df_new = pd.DataFrame(columns=[orient, 'Penetrance'])
    df_new[orient] = df[orient].astype(int)
    df_new['Penetrance'] = df['Penetrance'].astype(float)

    plt.figure(figsize=(20, 8))
    sns.set(font_scale=1.5)
    sns.set_style()
    cg = sns.violinplot(x=orient, y='Penetrance', data=df_new, color='gray')
    cg.set_title('Penetrance across the plate')
    plt.ylim(0, 100)
    fig = plt.gcf()
    fig.savefig(output, bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def PlotSortedPenetrance(df, wt_strains, output_file, pos_controls_df=np.array([])):
    """ Plots penetrance values for each well in a sorted manner, colored with controls. """

    orf = df['ORF'][df['Num_cells'] > 80].tolist()
    penetrance = df['Penetrance'][df['Num_cells'] > 80].tolist()
    strain = []
    l = len(orf)
    if len(pos_controls_df) > 0:
        for i in range(l):
            if orf[i] == wt_strains[0]:
                strain.append(1)
            elif orf[i] == wt_strains[1]:
                strain.append(2)
            elif orf[i] in pos_controls_df[pos_controls_df['Penetrance_bin'] == 0].ORF.tolist():
                strain.append(3)
            elif orf[i] in pos_controls_df[pos_controls_df['Penetrance_bin'] == 1].ORF.tolist():
                strain.append(4)
            elif orf[i] in pos_controls_df[pos_controls_df['Penetrance_bin'] == 2].ORF.tolist():
                strain.append(5)
            elif orf[i] in pos_controls_df[pos_controls_df['Penetrance_bin'] == 3].ORF.tolist():
                strain.append(6)
            else:
                strain.append(0)
        key = {0: ('white', 'Mutant'),
               1: ('red', 'WT - Column 1'),
               2: ('orange', 'WT - Column 2'),
               3: ('forestgreen', 'Bin 0'),
               4: ('dodgerblue', 'Bin 1'),
               5: ('magenta', 'Bin 2'),
               6: ('purple', 'Bin 3')}
    else:
        for i in range(l):
            if orf[i] == wt_strains[0]:
                strain.append(1)
            elif orf[i] == wt_strains[1]:
                strain.append(2)
            else:
                strain.append(0)
        key = {0: ('white', 'Mutant'),
               1: ('blue', 'WT - Column 1'),
               2: ('red', 'WT - Column 2')}

    colors = [key[index][0] for index in strain]

    plt.figure(figsize=(20, 8))
    plt.scatter(list(reversed(range(l))), penetrance, edgecolors='black', c=colors, s=40)
    plt.xlim([-5, l + 10])
    plt.ylim([-5, 105])
    plt.yticks([0, 25, 50, 75, 100])
    plt.xlabel('')
    plt.ylabel('Penetrance')
    plt.title('Penetrance across controls')

    # Plot legend
    patches = [matplotlib.patches.Patch(edgecolor='black', facecolor=color, label=label) for color, label in
               key.values()]
    fig = plt.gcf()
    plt.legend(handles=patches, labels=[label for _, label in key.values()], loc=2)
    fig.savefig(output_file, bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def PlotROCandPR(PC, WT, output1, output2, output3):
    """ Plots penetrance values for each well in a sorted manner, colored with controls. """

    tpr = []
    fpr = []
    prec = []
    for threshold in list(reversed(range(101))):
        tp = len(PC[PC >= threshold])
        fp = len(WT[WT >= threshold])
        tpr.append(tp / float(len(PC)))
        fpr.append(fp / float(len(WT)))
        if fp == 0:
            prec.append(1)
        else:
            prec.append(tp / float(tp + fp))
    auc = metrics.auc(fpr, tpr)

    # Save plot for ROC
    plt.figure(figsize=(6, 6))
    sns.set(font_scale=1.5)
    sns.set_style('white')
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUROC = %0.2f' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc='lower right')
    fig = plt.gcf()
    plt.savefig(output1)
    fig.clf()
    plt.close(fig)

    # Save plot for PR
    plt.figure(figsize=(6, 6))
    sns.set(font_scale=1.5)
    sns.set_style('white')
    plt.plot(tpr, prec, color='darkorange', lw=2)
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.02])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    fig = plt.gcf()
    plt.savefig(output2)
    fig.clf()
    plt.close(fig)

    # Save numbers
    roc_numbers = pd.DataFrame({'Penetrance_Cutoff': list(reversed(range(101))),
                                'TPR (Recall)': np.asarray(tpr),
                                'FPR': np.asarray(fpr),
                                'Precision': np.asarray(prec)})
    roc_numbers = roc_numbers[['Penetrance_Cutoff', 'TPR (Recall)', 'FPR', 'Precision']]
    roc_numbers = roc_numbers.sort_values('Penetrance_Cutoff', ascending=False)
    roc_numbers.to_csv(path_or_buf=output3, index=False)


def ConfusionMatrix(cm, classes, acc, label_actual, label_predicted, output_file):
    """ Produces confusion matrix for given classes. """

    # Normalize confusion matrix values
    plt.figure(figsize=(6, 6))
    sns.set(font_scale=1.5)
    sns.set_style()
    for i in range(len(cm)):
        cm[i] = np.array(list(reversed(cm[i])))
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Acc %.2f%%' % acc)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(list(reversed(tick_marks)), classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel(label_actual)
    plt.xlabel(label_predicted)
    plt.grid(False)
    fig = plt.gcf()
    plt.savefig(output_file, bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def GetPosControlFile(df, pos_controls_file, wt_strains, cell_threshold, output, output_files):
    """ Extracts data for PlotPerformance based on positive and negative controls. """

    # Positive controls
    pos_controls_df = pd.read_csv(pos_controls_file)
    pos_controls_df['Penetrance'] = np.copy(pos_controls_df['Row'])
    pos_controls_df['P-value'] = np.copy(pos_controls_df['Row'])
    pos_controls_df['Num_cells'] = np.copy(pos_controls_df['Row'])
    pos_controls_df['Predicted_Penetrance_bin'] = np.copy(pos_controls_df['Penetrance_bin'])
    for i in range(len(pos_controls_df)):
        g = pos_controls_df.iloc[i, pos_controls_df.columns.get_loc('Gene')]
        r = pos_controls_df.iloc[i, pos_controls_df.columns.get_loc('Row')]
        c = pos_controls_df.iloc[i, pos_controls_df.columns.get_loc('Column')]
        index = df[(df['Gene'] == g) & (df['Row'] == r) & (df['Column'] == c)].index.tolist()
        print("i", i, "g", g, "r", r, "c", c, "index", index, type(index))
        if g in df['Gene'].tolist():
            print("g", g)
            print("df", df)
            print(df.shape, df.columns)
            # Place penetrance, p value and predicted penetrance bins
            p = df.iloc[index[0], df.columns.get_loc('Penetrance')]
            pos_controls_df.iloc[i, pos_controls_df.columns.get_loc('Penetrance')] = p
            pval = df.iloc[index[0], df.columns.get_loc('P-value')]
            pos_controls_df.iloc[i, pos_controls_df.columns.get_loc('P-value')] = pval
            cellnum = df.iloc[index[0], df.columns.get_loc('Num_cells')]
            pos_controls_df.iloc[i, pos_controls_df.columns.get_loc('Num_cells')] = cellnum

            if p < 25:
                pos_controls_df.iloc[i, pos_controls_df.columns.get_loc('Predicted_Penetrance_bin')] = 0
            elif (p >= 25) and (p < 50):
                pos_controls_df.iloc[i, pos_controls_df.columns.get_loc('Predicted_Penetrance_bin')] = 1
            elif (p >= 50) and (p < 75):
                pos_controls_df.iloc[i, pos_controls_df.columns.get_loc('Predicted_Penetrance_bin')] = 2
            else:
                pos_controls_df.iloc[i, pos_controls_df.columns.get_loc('Predicted_Penetrance_bin')] = 3

        else:
            pos_controls_df.iloc[i, pos_controls_df.columns.get_loc('Penetrance')] = np.nan
            pos_controls_df.iloc[i, pos_controls_df.columns.get_loc('P-value')] = np.nan
            pos_controls_df.iloc[i, pos_controls_df.columns.get_loc('Num_cells')] = np.nan
            pos_controls_df.iloc[i, pos_controls_df.columns.get_loc('Predicted_Penetrance_bin')] = np.nan

    # Save the penetrance and predicted bins
    pos_controls_df = pos_controls_df.reset_index(drop=True)
    pos_controls_df = pos_controls_df[['ORF', 'Gene', 'Row', 'Column', 'Penetrance_bin', 'Predicted_Penetrance_bin',
                                       'Penetrance', 'P-value', 'Num_cells', 'Comments']]
    pos_controls_df.to_csv(path_or_buf=output[output_files[0]], index=False)
    PlotPenetranceAgreement(pos_controls_df, cell_threshold, output[output_files[1]])

    # Remove genes that are not screened for calculating accuracy
    pos_controls_df = pos_controls_df.dropna(axis=0, subset=['Penetrance', 'P-value', 'Predicted_Penetrance_bin'])
    pos_controls_df = pos_controls_df.reset_index(drop=True)

    # Accuracy
    actual = pos_controls_df.Penetrance_bin.tolist()
    predicted = pos_controls_df.Predicted_Penetrance_bin.tolist()

    # WT populations
    WT_penetrance = np.array(pos_controls_df[pos_controls_df['Penetrance_bin'] == 0].Penetrance)
    for w in wt_strains:
        WT_penetrance = np.append(WT_penetrance, np.asarray(df[df.ORF == w].Penetrance))

    # Positive Control populations
    PC_penetrance = np.array(pos_controls_df[(pos_controls_df['Penetrance_bin'] == 2) |
                                             (pos_controls_df['Penetrance_bin'] == 3)].Penetrance)

    # Plot sorted penetrance values
    PlotSortedPenetrance(df, wt_strains, output[output_files[2]], pos_controls_df)

    # Plot ROC
    PlotROCandPR(PC_penetrance, WT_penetrance, output[output_files[3]], output[output_files[4]], output[output_files[5]])

    return pos_controls_df, actual, predicted


def PlotPenetranceAgreement(df, cell_threshold, output):
    filtered_df = df[df['Num_cells'] >= cell_threshold]
    pene = filtered_df.Penetrance.tolist()
    bins = filtered_df['Penetrance_bin'].tolist()
    true_bins = []
    for b in bins:
        if b == 0:
            true_bins.append(12.5)
        elif b == 1:
            true_bins.append(37.5)
        elif b == 2:
            true_bins.append(62.5)
        elif b == 3:
            true_bins.append(87.5)

    plt.figure(figsize=(6, 6))
    sns.set(font_scale=1.5)
    sns.set_style()
    plt.scatter(pene, true_bins, marker='.', c='black', alpha=1)
    plt.xlim([-2, 102])
    plt.ylim([-2, 102])
    plt.xticks([0, 25, 50, 75, 100])
    plt.yticks([0, 25, 50, 75, 100])
    plt.xlabel('Calculated Penetrance')
    plt.ylabel('Penetrance Bin')
    plt.title('Penetrance Agreement')
    fig = plt.gcf()
    plt.savefig(output)
    fig.clf()
    plt.close(fig)


def PlotPerformance(screen_name, df, wt_strains, cell_threshold, output):
    """ Extracts screen positive control file, calculates ROC values and plots.
        Also plots the sorted penetrance values with positive and negative controls. """

    marker = screen_name.split('_')[3].upper()
    path = '/home/morphology/mpg4/dlo/bin/TerminalPhenotypes/Results/'
    pos_controls_file = path + marker + '/' + marker + '_TruePositives/' + screen_name + '_TruePositives.csv'
    pos_controls_file_DL = path + marker + '/' + marker + '_TruePositives/' + screen_name + '_TruePositives_DL.csv'
    pos_controls_file_HF = path + marker + '/' + marker + '_TruePositives/' + screen_name + '_TruePositives_HF.csv'
    print("screen_name", screen_name)
    print("marker", marker)
    print("path", path)
    print("pos_controls_file", pos_controls_file)
    print("pos_controls_file_DL", pos_controls_file_DL)
    print("pos_controls_file_HF", pos_controls_file_HF)
    # Requires positive control files from Dara Lo and Helena Friesen
    if os.path.isfile(pos_controls_file_DL) and os.path.isfile(pos_controls_file_HF):
        print('Plotting performances by Dara and Helena ...')

        output_files_DL = ['PenetranceBinsDL', 'PenetranceBinsPlotDL', 'PenetranceSortedDL', 'ROCCurveDL', 'PRCurveDL', 'CurveNumbersDL']
        output_files_HF = ['PenetranceBinsHF', 'PenetranceBinsPlotHF', 'PenetranceSortedHF', 'ROCCurveHF', 'PRCurveHF', 'CurveNumbersHF']

        # Positive controls
        pos_controls_df_DL, actual_DL, predicted_DL = GetPosControlFile(df, pos_controls_file_DL, wt_strains,
                                                                        cell_threshold, output, output_files_DL)

        pos_controls_df_HF, actual_HF, predicted_HF = GetPosControlFile(df, pos_controls_file_HF, wt_strains,
                                                                        cell_threshold, output, output_files_HF)

        # Calculate penetrance bin accuracy DL
        tp = 0
        for i in range(len(actual_DL)):
            if actual_DL[i] == predicted_DL[i]:
                tp += 1
        accuracy_DL = float(tp) / len(actual_DL)

        # Plot confusion matrix DL
        cm = metrics.confusion_matrix(actual_DL, predicted_DL)
        #classes = np.asarray(['Bin - 0', 'Bin - 1', 'Bin - 2', 'Bin - 3'])
        classes = np.asarray(['75-100', '50-75', '25-50', '0-25'])
        ConfusionMatrix(cm, classes, accuracy_DL, 'Dara Lo', 'Computer Prediction', output['ConfusionMatrixDL'])

        # Calculate penetrance bin accuracy HF
        tp = 0
        for i in range(len(actual_HF)):
            if actual_HF[i] == predicted_HF[i]:
                tp += 1
        accuracy_HF = float(tp) / len(actual_HF)

        # Plot confusion matrix HF
        cm = metrics.confusion_matrix(actual_HF, predicted_HF)
        #classes = np.asarray(['Bin - 0', 'Bin - 1', 'Bin - 2', 'Bin - 3'])
        classes = np.asarray(['75-100', '50-75', '25-50', '0-25'])
        ConfusionMatrix(cm, classes, accuracy_HF, 'Helena Friesen', 'Computer Prediction', output['ConfusionMatrixHF'])

        # Calculate penetrance bin accuracy DL and HF
        tp = 0
        for i in range(len(actual_DL)):
            if actual_DL[i] == actual_HF[i]:
                tp += 1
        accuracy = float(tp) / len(actual_DL)

        # Plot confusion matrix HF
        cm = metrics.confusion_matrix(actual_DL, actual_HF)
        #classes = np.asarray(['Bin - 0', 'Bin - 1', 'Bin - 2', 'Bin - 3'])
        classes = np.asarray(['75-100', '50-75', '25-50', '0-25'])
        ConfusionMatrix(cm, classes, accuracy, 'Dara Lo', 'Helena Friesen', output['ConfusionMatrixDLHF'])

    # Requires one positive controls file
    elif os.path.isfile(pos_controls_file):
        print('Plotting performances by Dara only...')

        output_files = ['PenetranceBins', 'PenetranceSorted', 'ROCCurve', 'PRCurve', 'CurveNumbers']

        # Positive controls
        pos_controls_df, actual, predicted = GetPosControlFile(df, pos_controls_file, wt_strains,
                                                               cell_threshold, output, output_files)

        # Calculate penetrance bin accuracy
        tp = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                tp += 1
        accuracy = float(tp) / len(actual)

        # Plot confusion matrix
        cm = metrics.confusion_matrix(actual, predicted)
        #classes = np.asarray(['Bin - 0', 'Bin - 1', 'Bin - 2', 'Bin - 3'])
        classes = np.asarray(['75-100', '50-75', '25-50', '0-25'])
        ConfusionMatrix(cm, classes, accuracy, 'Actual Label', 'Computer Prediction', output['ConfusionMatrix'])

    # No positive control file available
    else:
        # Plot sorted penetrance values
        PlotSortedPenetrance(df, wt_strains, output['PenetranceSorted'])
        print('No positive controls for this screen!')
