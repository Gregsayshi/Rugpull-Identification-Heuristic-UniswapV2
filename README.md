# Rugpull-Identification-Heuristic-UniswapV2
Retrospective heuristic to identify trading pairs that are subject to rug-pulls on Uniswap V2 (Master's Thesis)
>>>>>>>>> FOLDER STRUCTURE

\code
	\evaluation
		parameter_optimization_results_plotter.py		: generates plots visualizing the parameter optimization
		evaluator.py						: provides summary statistics on "rugpull_identification_results.csv"
		plotter.py						: creates plots from "rugpull_identification_results.csv"
		poolsize_plotter.py					: plots WETH poolsize development over time for target trading pair
	\parameter_optimization
		first_round_parameter_optimization.py  			: input parameter optimization with initial values
		second_round_parameter_optimization.py 			: input paramter optimization with promising values based on first round optimization results
	\rugpull_identification
		rugpull_analysis_optimized.py				: analysis of the overall data set with optimized input parameters
\data
	\base_dataset
		uniswap_data.zip					: zipped version of Uniswap V2 on-chain data for the period May 2020 - May 2021 (ATTENTION - NOT INCLUDED!!) 
	\computer_generated		
		\figures
			\parameter_optimization				: contains figures generated from results of parameter optimization process
			\rugpull_identification				: contains figures generated from results of rug-pull identification process	
		\parameter_optimization_results
			first_round_parameter_optimization_results.csv	: initial parameter optimization results
			second_round_parameter_optimization_results.csv : second round parameter optimization results
		\rugpull_identification_results
			rugpull_identification_results.csv		: results of the analysis of the overall data-set with optimized parameter settings
	\manually_generated		
		\ground_truth_samples
			Ground_Truth_Data.xlsx				: ground truth data set with evidence / proof of rug-pulls + dates (if applicable)
			Negative_Samples.txt				: non-rug-pull samples from ground truth dataset (pair addresses)
			Positive_Samples.txt				: rug-pull samples from ground truth dataset (pair addresses)
		Manual_Exam_Top_20_results.xlsx				: Manual Examination of top 20 (measured by number of sync events) attributed rug-pulls from "rugpull_analysis_optimized.py"


>>>>>>>>> REPRODUCTION OF RESULTS

(in all scripts: replace (YOUR_PATH) with your path)
01: unzip uniswap_database.zip into the same folder
02: run first_round_parameter_optimization.py
03: run second_round_parameter_optimization.py
04: run parameter_optimization_results_plotter.py for plot generation
05: run rugpull_analysis_optimized.py
06: run evaluator.py for summary stastistics
07: run plotter.py for plot generation
08: run poolsize_plotter.py for WETH poolsize over time plot generation (need to insert target trading pair address manually)
