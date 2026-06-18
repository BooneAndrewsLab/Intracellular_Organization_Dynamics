import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import polars as pl
from scipy import stats
from scipy.optimize import curve_fit
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--results_dir', default='', help='Path in which final OD outputs are stored and where output files will be written to.')

args = parser.parse_args()



### ==================== FUNCTIONS FOR COMBINING OD OUTPUT FILES AND CALCULATING PENETRANCE ====================
def merge_penetrance_files_per_rep(marker, results_dir):
    
    dfs = []
    for replicate in ["R1", "R2", "R3"]:
        rep_dfs = []
        for time_point in ["T00", "T03", "T06", "T09", "T12", "T15", "T18", "T21", "T24"]:
            file_path = f"{str(next(Path(results_dir).rglob(f"{marker.upper()}_{replicate}")))}/*_{marker.title()}_{replicate}_{time_point}_OD_results.csv"
            try:
                df = (
                    pl
                    .read_csv(file_path)
                    .with_columns(
                        pl.lit(replicate).alias("Rep"),
                        pl.lit(time_point).alias("Timepoint")
                    )
                    .select(['ORF', 'Gene', 'Penetrance', 'P-value', 'Num_cells', 'Timepoint', 'Rep', 'Row', 'Column'])
                )
                rep_dfs.append(df)
            except:
                continue
        
        # Combine and save dataframes for a single replicate
        rep_df = (
            pl
            .concat(rep_dfs, how="vertical")
            .with_columns(pl.lit(marker).alias("Marker"))
            )
        dfs.append(rep_df)
        
    
    # Combine and save dataframes for all replicates
    combined_info_df = (
        pl
        .concat(dfs, how="vertical")
        .sort(["Marker", "Gene", "Timepoint", "Rep"])
    )
    return combined_info_df


def calculate_combined_penenetrance(marker, combined_info_df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    aggregated_df = (
        combined_info_df
        .group_by("Marker", "ORF", "Gene", "Timepoint")
        .agg(
            ((pl.col("Penetrance") * pl.col("Num_cells")).sum() / pl.col("Num_cells").sum()).alias("Penetrance"),
            pl.col("Num_cells").sum().alias("Num_cells")
        )
        .sort(["Marker", "Gene", "Timepoint"])
    )
    
    aggregated_df.write_csv(f"{output_dir}/{marker}_Overall_Penetrance_CellNum.csv")
    

def get_all_marker_penetrances_spreadsheet(combined_info_df, combined_dir):
    
    # Get all per-rep data and rename each WT technical replicate
    per_rep_data = (
        combined_info_df
        .select(["Rep", "Marker", "ORF", "Row", "Column", "Gene", "Timepoint", "Penetrance", "Num_cells"])
        .rename({"Num_cells": "Cell_Count", "ORF": "Strain"})
        .with_columns(
            (
                pl
                .when(pl.col("Strain") == "YOR202W")
                .then(
                    pl.concat_str(
                        [
                            pl.lit("WT1"),
                            pl.col("Row").cast(pl.Utf8),
                            pl.col("Column").cast(pl.Utf8)
                        ], separator="_",
                    )
                )
                .when(pl.col("Strain") == "WT")
                .then(
                    pl.concat_str(
                        [
                            pl.lit("WT2"),
                            pl.col("Row").cast(pl.Utf8),
                            pl.col("Column").cast(pl.Utf8)
                        ], separator="_",
                    )
                )
                .otherwise(pl.col("Strain"))
                ).alias("Strain")
            )
        )


    # ==================================================================================================

    # Load pre-calculated overall penetrances but remove WT (they will be re-calculated to account for 
    # technical replicates)
    aggregated_data_mutant = (
        pl
        .read_csv(f"{combined_dir}/*.csv")
        .rename({"Penetrance": "Overall_Penetrance", "Num_cells": "Total_Cell_Count", "ORF": "Strain"})
        .filter(pl.col("Gene") != "WT")  # WT technical reps will be calculated separately
        )

    aggregated_data_wt = (
        per_rep_data
        .filter(pl.col("Gene") == "WT")
        .with_columns(
            ((pl.col("Penetrance") / 100) * pl.col("Cell_Count")).round(0).alias("Num_Outliers")
        )
        .drop("Penetrance")
        .group_by(["Marker", "Strain", "Row", "Column", "Gene", "Timepoint"])
        .agg(
            pl.col("Cell_Count").sum().alias("Total_Cell_Count"),
            pl.col("Num_Outliers").sum()
        )
        .with_columns(
            (pl.col("Num_Outliers") / pl.col("Total_Cell_Count") * 100).alias("Overall_Penetrance")
        )
        .drop("Num_Outliers")
        .select(aggregated_data_mutant.columns)
    )

    aggregated_data = pl.concat([aggregated_data_mutant, aggregated_data_wt], how="vertical").sort(["Marker", "Timepoint", "Strain"])

    # ==================================================================================================

    # Combine per-rep info and overall penetrances
    per_rep_data_wide = (
        per_rep_data
        .pivot(
            index=["Marker", "Strain", "Gene", "Timepoint"],
            columns="Rep",
            values=["Penetrance", "Cell_Count"]
            )
        .sort(["Marker", "Timepoint", "Strain"])
    )

    (
        aggregated_data
        .join(per_rep_data_wide, on=["Marker", "Strain", "Gene", "Timepoint"], how="left")
        .sort(["Marker", "Timepoint"])
        .write_csv(f"{combined_dir}/combined_marker_information_from_all_reps.csv"))
    

### ==================== FUNCTIONS FOR HALF MAX CALCULATIONS AND PLOTS ====================
def subset_gene_marker(df, gene, column):
    marker_df = df.filter(pl.col("Gene") == gene)
    
    times = np.array([int(tp[1:]) for tp in marker_df["Timepoint"]])
    pen_or_cells = marker_df[column].to_numpy()
    
    # Mask out NaN penetrance values
    mask = ~np.isnan(pen_or_cells)
    times = times[mask]
    pen_or_cells = pen_or_cells[mask]
    
    return times, pen_or_cells

def sigmoid_func(x, inflect, max_f, min_f, slope_f):
    return max_f / (1 + np.exp(-slope_f * (x - inflect))) + min_f

def sigmoid_func_deriv(x, inflect, max_f, min_f, slope_f):
    return max_f * slope_f * (np.exp(-slope_f * (x - inflect))) / (1 + np.exp(-slope_f * (x - inflect)))**2

def sigmoid_func_inverse(x, inflect, max_f, min_f, slope_f):
    if (x - min_f) < 0:
        return np.nan
    else:
        return inflect - math.log(max_f / (x - min_f) - 1.0) / slope_f

def sigmoid_func_lag(inflect, max_f, min_f, slope_f):
    popt = inflect, max_f, min_f, slope_f
    return inflect + (sigmoid_func(0, *popt) - sigmoid_func(inflect, *popt)) / sigmoid_func_deriv(inflect, *popt)

def sigmoid_func_char(time, penetrance, error):
    char, _ = curve_fit(f=sigmoid_func, xdata=time, ydata=penetrance, sigma=error,
                        bounds=([0,-np.inf,-np.inf,0], [24,np.inf,np.inf,np.inf]), method='trf')
    half_max = char[0]
    maxx = sigmoid_func(24, *char)
    minn = sigmoid_func(0, *char)
    half_max_calc = sigmoid_func_inverse(np.mean([maxx, minn]), *char)
    half_max_calc_min0 = sigmoid_func_inverse(np.mean([maxx, 0]), *char)
    slope = sigmoid_func_deriv(half_max, *char)
    lag = sigmoid_func_lag(*char)
    R2 = r2_score(penetrance, sigmoid_func(time, *char))
    
    return half_max, half_max_calc, half_max_calc_min0, maxx, minn, slope, lag, R2, char

def gpr_smoothing(X_original, y_original):
    np.random.seed(1)

    X = X_original.reshape(-1,1)
    y = np.array(y_original, dtype=float)
    x = np.atleast_2d(np.linspace(0, 24, 1000)).T
    
    dy = 0.5 + 1.0 * np.random.random(y.shape)
    noise = np.random.normal(0, dy)
    y = y.astype(float) + noise
    alpha=(dy / y) ** 2

    # Instantiate a Gaussian Process model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-3, 1e3))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x, return_std=True)
    LL = gp.log_marginal_likelihood()
    
    return x.reshape(-1), y_pred.reshape(-1), -LL, sigma

def subset_gpr_smoothing(time, penetrance):
    subset_penetrance = []
    for t in [0,3,6,9,12,15,18,21,24]:
        index = np.argmin(abs(time - t))
        subset_penetrance.append(penetrance[index])
    
    return subset_penetrance

def fit_curve_gene_marker(gene, marker_df):
    time_9, penetrance_9 = subset_gene_marker(marker_df, gene, "Penetrance")
    values = tuple(np.repeat(np.nan, 5))
    func_values = tuple(np.repeat(np.nan, 8))
    char = tuple(np.repeat(np.nan, 4))
    
    if len(time_9) > 3:
        time, penetrance, _, sigma = gpr_smoothing(time_9, penetrance_9)
        error = np.ones(len(penetrance))
        time_c, cellnum = subset_gene_marker(marker_df, gene, "Num_cells")
        _, cellnum, _, _ = gpr_smoothing(time_c, cellnum)
        error = 1.0 / np.sqrt(cellnum)
        error = [np.nanmax(error) if np.isnan(x) else x for x in error]

        try:
            half_max, half_max_calc, half_max_calc_min0, maxx, minn, slope, lag, R2, char = sigmoid_func_char(time, penetrance, error)
        except RuntimeError:
            half_max, half_max_calc, half_max_calc_min0, maxx, minn, slope, lag, R2 = tuple(np.repeat(np.nan, 8))
            char = tuple(np.repeat(np.nan, 4))
            
        if (not np.isnan(slope)) and (slope > 0) and (R2 > 0):
            values = (time_9, penetrance_9, time, penetrance, sigma)
            func_values = (half_max, half_max_calc, half_max_calc_min0, maxx, minn, slope, lag, R2)
        else:
            values = tuple(np.repeat(np.nan, 5))
            func_values = tuple(np.repeat(np.nan, 8))
            char = tuple(np.repeat(np.nan, 4))

    return values, func_values, char


def make_halfmax_plots(marker, input_dir, output_dir):
    marker_df = (
        pl
        .read_csv(f"{input_dir}/{marker}_Overall_Penetrance_CellNum.csv")
        .filter(
            pl.col("Num_cells") >= 50,
            pl.col("Gene") != "WT"
            )
        )
    genes = marker_df["Gene"].unique().to_list()

    ouput_plots_folder = f"{output_dir}/{marker}/gene_rep_fit_plots"

    if not os.path.exists(ouput_plots_folder):
        os.makedirs(ouput_plots_folder)

    OUT_DF = pd.DataFrame(columns = ['Gene', 'Marker', 'Half_max', 'Half_max_calc', 'Half_max_calc_min0',
                                     'Max', 'Min', 'Slope', 'Lag', 'R2',
                                     'Func_inflect', 'Func_max', 'Func_min', 'Func_slope'])
    this_row = 0
    for gene in genes:
        i=0
        plt.figure(figsize=(25, 25))

        values, func_values, char = fit_curve_gene_marker(gene, marker_df)
        time_9, penetrance_9, time, penetrance, sigma = values
        half_max, half_max_calc, half_max_calc_min0, maxx, minn, slope, lag, R2 = func_values
        inflect, max_f, min_f, slope_f = char
        OUT_DF.loc[this_row, ] = [gene, marker, half_max, half_max_calc, half_max_calc_min0,
                                  maxx, minn, slope, lag, R2,
                                  inflect, max_f, min_f, slope_f]
        this_row += 1

        # Plot
        i+=1
        plot_title = f"{marker} - {gene}"
        if not np.isnan(time_9).any():
            # Plot the function, the prediction and the 95% confidence interval based on the MSE
            plt.subplot(5, 5, i)
            plt.plot(0,0, 'w.', label='$R2$: %.4f' % R2)
            plt.plot(time, sigmoid_func(time, *char), 'r:', label='Logistic fit')
            plt.plot(time_9, penetrance_9, 'r.', markersize=15)
            plt.plot(half_max_calc, sigmoid_func(half_max_calc, *char), 'k*', markersize=15, label='Half-max time')
            plt.plot(time, penetrance, 'b-')
            plt.fill(np.concatenate([time, time[::-1]]),
                     np.concatenate([penetrance - 1.9600 * sigma,
                                    (penetrance + 1.9600 * sigma)[::-1]]), alpha=.5, fc='b', ec='None')
            plt.title(plot_title)
            plt.xlabel('Time')
            plt.ylabel('Penetrance')
            plt.ylim(-0.5, 120)
            plt.xticks([0,3,6,9,12,15,18,21,24])
            plt.xlim(-0.5, 24.5)
            plt.legend(loc='lower right')

        else:
            plt.subplot(5, 5, i)
            plt.title(plot_title)
            plt.xlabel('Time')
            plt.ylabel('Penetrance')
            plt.ylim(-0.5, 120)
            plt.xticks([0,3,6,9,12,15,18,21,24])
            plt.xlim(-0.5, 24.5)

        fig = plt.gcf()
        fig.savefig(f"{ouput_plots_folder}/GPR_noise_fitcell_{gene}.png", bbox_inches='tight')
        plt.close(fig)
        OUT_DF.to_csv(f"{output_dir}/{marker}/Logistic_curve_fitting_cellnum_GPR_noise.csv", index = False)
        

if __name__ == '__main__':
    #markers = [
    #    "Atg8", "Cdc11", "Dad2", "Dcp2", 
    #    "Heh2", "Mca1", "Mid2", "Nhp6a", 
    #    "Nop10", "Nuf2", "Om45", "Pil1", 
    #    "Pre2", "Rad52", "Sac6", "Sec7", 
    #    "Sec21", "Snf7", "Spf1", "Tgl3", 
    #    "Vph1"
    #    ]

    markers = ["Atg8"] # Only Atg8 screen here for demo purposes -- the commented out list above includes all screens used in the study
    results_dir = args.results_dir

    
    # Getting final penetrance calculations
    for marker in markers:
        combined_info_df = merge_penetrance_files_per_rep(
            marker=marker, 
            results_dir=f"{results_dir}/OD_results")
        
        calculate_combined_penenetrance(
            marker=marker, 
            combined_info_df=combined_info_df,
            output_dir=f"{results_dir}/combined_output_results")

    get_all_marker_penetrances_spreadsheet(
        combined_info_df=combined_info_df,
        combined_dir=f"{results_dir}/combined_output_results"
        )
    
    # Getting half-max calculations and plots
    for marker in markers:
        make_halfmax_plots(
            marker=marker,
            input_dir=f"{results_dir}/combined_output_results",
            output_dir=f"{results_dir}/half_max_analysis"
            )