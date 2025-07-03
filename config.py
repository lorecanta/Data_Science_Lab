# This file contains the configuration for the data processing and analysis of the survey data.

knowledge_score_variables = ["qk3_score", "qk4_score", "qk5_score", "qk6_score", "qk7_1_score", "qk7_2_score", "qk7_3_score"]
behavioral_score_variables = ["qf1_qf2_score", "qf3_score", "qf10_1_score", "qf10_4_score", "qf10_6_score", "qf10_7_score", "qprod_2pt_score", "qprod_1pt_score", "qf12_score"]
attitude_score_variables = ["qf10_2_score", "qf10_3_score", "qf10_8_score"]

min_weight = 200
n_folds = 10

data_path = "data/Financia_literacy_2018.csv"