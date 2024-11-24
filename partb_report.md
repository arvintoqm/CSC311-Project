# CSC311 Report

**1**, From Part 1, we figured out that the logistic regression ensemble has the best performance overall. But, in part 1, we only used the primary data and other meta-datas are not utilized. There might be a meaningful representation if we combine these datasets

(papers) leelab-africanllm-2@csgpu:~/AfricanNLP/src/final_project$ python [ensemble.py](http://ensemble.py/)
Model 1 trained.
Model 2 trained.
Model 3 trained.
Validation Accuracy: 0.7044877222692634
Test Accuracy: 0.707310189105278

(From part 1 Ensemble)

**2**, I designed a combined dataset pipeline with ensemble logistic regression. Based on this finding, we can conclude that combined datasets improve the performance than using only the primary data

(papers) leelab-africanllm-2@csgpu:~/AfricanNLP/src/final_project$ python ensemble_all_features.py
Validation Accuracy: 0.7049

**3**, Since there might be other methodologies that can capture representation patterns in a different way than logistic regressions, we tried other methods, IRT, other recipes for ensemble, neural nets, and matrix factorization. Based on the experiment, we see IRT has competitive performance with 3 logistic ensemble. NOTE: I've not tested auto-encoders and other types of neural nets. We might have better results with different architectures.  

(papers) leelab-africanllm-2@csgpu:~/AfricanNLP/src/final_project$ python all_models_features.py
Ensemble Validation Accuracy: 0.6722
Iteration 1, Validation Accuracy: 0.6908
Iteration 2, Validation Accuracy: 0.6948
Iteration 3, Validation Accuracy: 0.6980
Iteration 4, Validation Accuracy: 0.7012
Iteration 5, Validation Accuracy: 0.7031
Iteration 6, Validation Accuracy: 0.7065
Iteration 7, Validation Accuracy: 0.7089
Iteration 8, Validation Accuracy: 0.7077
Iteration 9, Validation Accuracy: 0.7060
Iteration 10, Validation Accuracy: 0.7070
IRT Validation Accuracy: 0.7070
Epoch [1/10], Validation Accuracy: 0.6709
Epoch [2/10], Validation Accuracy: 0.6712
Epoch [3/10], Validation Accuracy: 0.6812
Epoch [4/10], Validation Accuracy: 0.6681
Epoch [5/10], Validation Accuracy: 0.6812
Epoch [6/10], Validation Accuracy: 0.6792
Epoch [7/10], Validation Accuracy: 0.6739
Epoch [8/10], Validation Accuracy: 0.6785
Epoch [9/10], Validation Accuracy: 0.6806
Epoch [10/10], Validation Accuracy: 0.6809
Matrix Factorization Validation Accuracy: 0.4537

**4**, Now, we have figured out that IRT and logistic regression are two top-performance models. Let's combine these two models into one ensemble. The result were slightly less than only the logistic regression model

(papers) leelab-africanllm-2@csgpu:~/AfricanNLP/src/final_project$ python ensemble_logistic_irt.py
Validation Accuracy of Ensemble: 0.7062
Iteration 1, Validation Accuracy: 0.6908
Iteration 2, Validation Accuracy: 0.6948
Iteration 3, Validation Accuracy: 0.6980
Iteration 4, Validation Accuracy: 0.7012
Iteration 5, Validation Accuracy: 0.7031
Iteration 6, Validation Accuracy: 0.7065
Iteration 7, Validation Accuracy: 0.7089
Iteration 8, Validation Accuracy: 0.7077
Iteration 9, Validation Accuracy: 0.7060
Iteration 10, Validation Accuracy: 0.7070
IRT Validation Accuracy: 0.7070
Combined Validation Accuracy: 0.7069

**5**, It seems better to focus only on the logistic regression ensemble. The remaining consideration is the feature choices. 

(papers) leelab-africanllm-2@csgpu:~/AfricanNLP/src/final_project$ python ensemble_select_features.py
Features: User_feat=True, Question_feat=True, Additional_features=[], Val_accuracy=0.7007
Features: User_feat=True, Question_feat=True, Additional_features=['age'], Val_accuracy=0.7038
Features: User_feat=True, Question_feat=True, Additional_features=['premium_pupil'], Val_accuracy=0.7067
Features: User_feat=True, Question_feat=True, Additional_features=['student_avg_correct'], Val_accuracy=0.7083
Features: User_feat=True, Question_feat=True, Additional_features=['reduced_embedding'], Val_accuracy=0.7058
Features: User_feat=True, Question_feat=True, Additional_features=['age', 'premium_pupil'], Val_accuracy=0.7052
Features: User_feat=True, Question_feat=True, Additional_features=['age', 'student_avg_correct'], Val_accuracy=0.7046
Features: User_feat=True, Question_feat=True, Additional_features=['age', 'reduced_embedding'], Val_accuracy=0.7058
Features: User_feat=True, Question_feat=True, Additional_features=['premium_pupil', 'student_avg_correct'], Val_accuracy=0.7100
Features: User_feat=True, Question_feat=True, Additional_features=['premium_pupil', 'reduced_embedding'], Val_accuracy=0.7056
Features: User_feat=True, Question_feat=True, Additional_features=['student_avg_correct', 'reduced_embedding'], Val_accuracy=0.7058
Features: User_feat=True, Question_feat=True, Additional_features=['age', 'premium_pupil', 'student_avg_correct'], Val_accuracy=0.7048
Features: User_feat=True, Question_feat=True, Additional_features=['age', 'premium_pupil', 'reduced_embedding'], Val_accuracy=0.7058
Features: User_feat=True, Question_feat=True, Additional_features=['age', 'student_avg_correct', 'reduced_embedding'], Val_accuracy=0.7062
Features: User_feat=True, Question_feat=True, Additional_features=['premium_pupil', 'student_avg_correct', 'reduced_embedding'], Val_accuracy=0.7103
Features: User_feat=True, Question_feat=True, Additional_features=['age', 'premium_pupil', 'student_avg_correct', 'reduced_embedding'], Val_accuracy=0.7063
Features: User_feat=True, Question_feat=False, Additional_features=[], Val_accuracy=0.6820
Features: User_feat=True, Question_feat=False, Additional_features=['age'], Val_accuracy=0.6775
Features: User_feat=True, Question_feat=False, Additional_features=['premium_pupil'], Val_accuracy=0.6802
Features: User_feat=True, Question_feat=False, Additional_features=['student_avg_correct'], Val_accuracy=0.6823
Features: User_feat=True, Question_feat=False, Additional_features=['reduced_embedding'], Val_accuracy=0.6815
Features: User_feat=True, Question_feat=False, Additional_features=['age', 'premium_pupil'], Val_accuracy=0.6802
Features: User_feat=True, Question_feat=False, Additional_features=['age', 'student_avg_correct'], Val_accuracy=0.6811
Features: User_feat=True, Question_feat=False, Additional_features=['age', 'reduced_embedding'], Val_accuracy=0.6788
Features: User_feat=True, Question_feat=False, Additional_features=['premium_pupil', 'student_avg_correct'], Val_accuracy=0.6775
Features: User_feat=True, Question_feat=False, Additional_features=['premium_pupil', 'reduced_embedding'], Val_accuracy=0.6833
Features: User_feat=True, Question_feat=False, Additional_features=['student_avg_correct', 'reduced_embedding'], Val_accuracy=0.6797
Features: User_feat=True, Question_feat=False, Additional_features=['age', 'premium_pupil', 'student_avg_correct'], Val_accuracy=0.6801
Features: User_feat=True, Question_feat=False, Additional_features=['age', 'premium_pupil', 'reduced_embedding'], Val_accuracy=0.6802
Features: User_feat=True, Question_feat=False, Additional_features=['age', 'student_avg_correct', 'reduced_embedding'], Val_accuracy=0.6791
Features: User_feat=True, Question_feat=False, Additional_features=['premium_pupil', 'student_avg_correct', 'reduced_embedding'], Val_accuracy=0.6825
Features: User_feat=True, Question_feat=False, Additional_features=['age', 'premium_pupil', 'student_avg_correct', 'reduced_embedding'], Val_accuracy=0.6811
Features: User_feat=False, Question_feat=True, Additional_features=[], Val_accuracy=0.6188
Features: User_feat=False, Question_feat=True, Additional_features=['age'], Val_accuracy=0.6266
Features: User_feat=False, Question_feat=True, Additional_features=['premium_pupil'], Val_accuracy=0.6208
Features: User_feat=False, Question_feat=True, Additional_features=['student_avg_correct'], Val_accuracy=0.7069
Features: User_feat=False, Question_feat=True, Additional_features=['reduced_embedding'], Val_accuracy=0.6219
Features: User_feat=False, Question_feat=True, Additional_features=['age', 'premium_pupil'], Val_accuracy=0.6201
Features: User_feat=False, Question_feat=True, Additional_features=['age', 'student_avg_correct'], Val_accuracy=0.7076
Features: User_feat=False, Question_feat=True, Additional_features=['age', 'reduced_embedding'], Val_accuracy=0.6197
Features: User_feat=False, Question_feat=True, Additional_features=['premium_pupil', 'student_avg_correct'], Val_accuracy=0.7049
Features: User_feat=False, Question_feat=True, Additional_features=['premium_pupil', 'reduced_embedding'], Val_accuracy=0.6273
Features: User_feat=False, Question_feat=True, Additional_features=['student_avg_correct', 'reduced_embedding'], Val_accuracy=0.7077
Features: User_feat=False, Question_feat=True, Additional_features=['age', 'premium_pupil', 'student_avg_correct'], Val_accuracy=0.7038
Features: User_feat=False, Question_feat=True, Additional_features=['age', 'premium_pupil', 'reduced_embedding'], Val_accuracy=0.6249
Features: User_feat=False, Question_feat=True, Additional_features=['age', 'student_avg_correct', 'reduced_embedding'], Val_accuracy=0.7072
Features: User_feat=False, Question_feat=True, Additional_features=['premium_pupil', 'student_avg_correct', 'reduced_embedding'], Val_accuracy=0.7099
Features: User_feat=False, Question_feat=True, Additional_features=['age', 'premium_pupil', 'student_avg_correct', 'reduced_embedding'], Val_accuracy=0.7072
Features: User_feat=False, Question_feat=False, Additional_features=['age'], Val_accuracy=0.5986
Features: User_feat=False, Question_feat=False, Additional_features=['premium_pupil'], Val_accuracy=0.6008
Features: User_feat=False, Question_feat=False, Additional_features=['student_avg_correct'], Val_accuracy=0.6840
Features: User_feat=False, Question_feat=False, Additional_features=['reduced_embedding'], Val_accuracy=0.6008
Features: User_feat=False, Question_feat=False, Additional_features=['age', 'premium_pupil'], Val_accuracy=0.6008
Features: User_feat=False, Question_feat=False, Additional_features=['age', 'student_avg_correct'], Val_accuracy=0.6840
Features: User_feat=False, Question_feat=False, Additional_features=['age', 'reduced_embedding'], Val_accuracy=0.5986
Features: User_feat=False, Question_feat=False, Additional_features=['premium_pupil', 'student_avg_correct'], Val_accuracy=0.6840
Features: User_feat=False, Question_feat=False, Additional_features=['premium_pupil', 'reduced_embedding'], Val_accuracy=0.6008
Features: User_feat=False, Question_feat=False, Additional_features=['student_avg_correct', 'reduced_embedding'], Val_accuracy=0.6832
Features: User_feat=False, Question_feat=False, Additional_features=['age', 'premium_pupil', 'student_avg_correct'], Val_accuracy=0.6839
Features: User_feat=False, Question_feat=False, Additional_features=['age', 'premium_pupil', 'reduced_embedding'], Val_accuracy=0.5986
Features: User_feat=False, Question_feat=False, Additional_features=['age', 'student_avg_correct', 'reduced_embedding'], Val_accuracy=0.6836
Features: User_feat=False, Question_feat=False, Additional_features=['premium_pupil', 'student_avg_correct', 'reduced_embedding'], Val_accuracy=0.6830
Features: User_feat=False, Question_feat=False, Additional_features=['age', 'premium_pupil', 'student_avg_correct', 'reduced_embedding'], Val_accuracy=0.6840

Best feature combination:
{'include_user_feat': True, 'include_question_feat': True, 'feature_list': ['premium_pupil', 'student_avg_correct', 'reduced_embedding'], 'val_accuracy': 0.7102737792830934}

Final Validation Accuracy with Best Features: 0.7067