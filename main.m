%%% Parameters to be adjusted：
% anchor_num; Number of anchors
% k; Number of nearest neighbor connection anchors
% beta; Hyperparameters of the tensor alignment term
% delta; Hyperparameters for control matrix invertibility
% order; % The highest order of bipartite graph

clear;clc;

addpath([pwd, '/funs']); 
addpath([pwd, '/measure']);
addpath([pwd, '/CLR']);
addpath([pwd, '/disrupt_index']);
addpath([pwd, '/datasets']);

% datasets
datasets = {'ORL','Caltech101-7'};
datapath = fullfile(pwd, 'datasets/');



%% Load data and parameter settings
for datasets_i = 1:1 
    
    result_all = zeros(10,8);
    time_all = zeros(10,1);

    for disrupt_data_ratio = 0.5:0.5:1
    
        for disrupt_i = 1:10
    
            clearvars -except datasets datapath datasets_i result_all time_all disrupt_data_ratio disrupt_i 
    
            dataname = datasets{datasets_i};
            load(strcat(datapath,dataname,'.mat'));
            disrupt_index_name = strcat(dataname,'_',num2str(disrupt_data_ratio),'.mat');
            load(disrupt_index_name);
    
            X = M;
            y = gnd; 
            views_num = length(X);
            cluster_num = length(unique(y));

            % Load disruption factor
            disrupt_index_all = disrupt_index_all_10{disrupt_i};

            % Parameter settings
            isGraph = 0; % Input is not a graph
            select_best_view_style = 2; % Most reliable view selection methods, 1: kmeans, 2: CLR

            anchor_num = 100; % Number of anchors
            k = 10;  % Number of nearest neighbor connection anchors
            optimize_iter_P = 20; % The maximum number of iterations for optimizing P
            optimize_iter_SF = 100; % The maximum number of iterations for optimizing SF
            delta = 1; % Hyperparameters for control matrix invertibility
            beta = 1; % Hyperparameters of the tensor alignment term
            order = 3; % The highest order of bipartite graph
    
            if cluster_num >= 10
                anchor_num = 200;
            end

            tic;
    
    
    
            %% Clustering
            % Calling the Cluster function for clustering
            [y_pred, y, best_view, d_P, iter_P] = Cluster(dataname, X, y, cluster_num, views_num, order, beta, isGraph, select_best_view_style, anchor_num, k, optimize_iter_P, optimize_iter_SF, delta, disrupt_index_all);    
    


            %% Calculate the clustering results
            result = zeros(1,8);
            [result(1,:)] = Clustering_Measure(y, y_pred); 
            result_all(disrupt_i,:) = result;
            time = toc; 
            time_all(disrupt_i) = time;
    

    
            %% Output clustering results
            % Direct Outputs
            fprintf("ACC nmi Purity Fscore Precision Recall AR Entropy\n");
            fprintf('%.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\nDataset:%s\nTime:%.4f\n\n',result(1),result(2),result(3),result(4),result(5),result(6),result(7),result(8),dataname,time);
    
            fprintf('disrupt_data_ratio:%.2f\n',disrupt_data_ratio);
            fprintf('anchor_num:%.3f\n',anchor_num);
            fprintf('k:%d\n',k);
            fprintf('delta:%.2f\n',delta);
            fprintf('best_view:%d\n\n\n\n',best_view);
    
            % Output to file
            fid = fopen('allresult.txt','a');
            fprintf(fid,"ACC nmi Purity Fscore Precision Recall AR Entropy\n");
            fprintf(fid,'%.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\nDataset:%s\nTime:%.4f\n\n',result(1),result(2),result(3),result(4),result(5),result(6),result(7),result(8),dataname,time);
    
            fprintf(fid,'disrupt_data_ratio:%.2f\n',disrupt_data_ratio);
            fprintf(fid,'anchor_num:%.3f\n',anchor_num);
            fprintf(fid,'k:%d\n',k);
            fprintf(fid,'delta:%.2f\n',delta);
            fprintf(fid,'beta:%.2f\n',beta);
            fprintf(fid,'best_view:%d\n\n\n\n',best_view);
    
            fclose(fid);
            
        end
    
        % Calculate mean and variance
        fprintf('disrupt_data_ratio:%.2f\n',disrupt_data_ratio);
        fprintf('anchor_num:%.3f\n',anchor_num);
        fprintf('k:%d\n',k);
        fprintf('delta:%.2f\n',delta);
        fprintf('beta:%.2f\n',beta);
        fprintf('best_view:%d\n\n',best_view);
        fprintf("Mean and variance\n");
        fprintf("ACC nmi Purity Fscore Precision Recall AR Entropy\n");
        fprintf('%.6f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\n',mean(result_all(1:disrupt_i,:),1)); 
        fprintf('%.6f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\n',var(result_all(1:disrupt_i,:),1)); 
        fprintf("Time\n");
        fprintf('%.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\n\n\n\n\n\n\n\n\n\n',sum(time_all),time_all); 
    
        fid = fopen('allresult.txt','a');
        fprintf(fid,'disrupt_data_ratio:%.2f\n',disrupt_data_ratio);
        fprintf(fid,'anchor_num:%.3f\n',anchor_num);
        fprintf(fid,'k:%d\n',k);
        fprintf(fid,'delta:%.2f\n',delta);
        fprintf(fid,'beta:%.2f\n',beta);
        fprintf(fid,'best_view:%d\n\n',best_view);
        fprintf(fid,"Mean and variance\n");
        fprintf(fid,"ACC nmi Purity Fscore Precision Recall AR Entropy\n");
        fprintf(fid,'%.6f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\n',mean(result_all(1:disrupt_i,:),1)); 
        fprintf(fid,'%.6f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\n',var(result_all(1:disrupt_i,:),1)); 
        fprintf(fid,"Time\n");
        fprintf(fid,'%.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\n\n\n\n\n\n\n\n\n\n',sum(time_all),time_all); 
        fclose(fid);
    
    end
    
end