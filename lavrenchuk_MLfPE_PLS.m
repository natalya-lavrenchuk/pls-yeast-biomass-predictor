

% Name: Natalya Lavrenchuk (ID: 2141882)
% Course: SP25 Machine Learning for Process Engineering
% Instructor: Professor Pierantonio Facco
% Assignment: Homework #1 – PLS Model
% Date: May 26, 2025

% The purpose of this assignment is to build a Partial Least Squares (PLS) 
% regression model to predict the final biomass concentration in a batch 
% yeast cultivation process based on historical online process data. 
% MATLAB PLS_Toolbox was used to construct the model and to answer the 
% questions on model fit and predictability.

clc
clear all
close all

load('Users/natalya/Documents/MATLAB/MLfPE/data/dataset_pls.mat')


%% ---- Question 1 - data visualization
var_names = {'glucose', 'pyruvate', 'acetaldehyde', 'acetate', 'ethanol', 'active cells', 'protein activity'};

for v=1:7
    a(:,:)= X3Dc(:,v,:);
    figure;
    plot(a');
    xlabel('time');
    ylabel(['variable #' num2str(v)]);
    box on
end

boxplot(Yc);
figure;plot(Yc);xlabel('observation');ylabel('end-point biomass concentration')


%% ---- Question 2 - Build the PLS model

% Reshape to 2D: batch-wise unfolding
X2Dc = reshape(X3Dc, size(X3Dc,1), []);  
X2Dv = reshape(X3Dv, size(X3Dv,1), []);

% PLS model building + autoscaling
o.display='off';
o.plots='none';
o.preprocessing={'autoscale' 'autoscale'}; %autoscale the data
plsm=pls(X2Dc,Yc,4,o); % Select 4LVs based on RMSEC
PLSm=plsm;

% Building PLStable
% Autoscale X and Y
X_scaled = auto(X2Dc);
Y_scaled = auto(Yc);

% Extract PLS model scores and loadings
T = plsm.T;  % X scores
P = plsm.P;  % X loadings
U = plsm.U;  % Y scores
Q = plsm.Q;  % Y loadings
W = plsm.wts;         % Weights 

numLVs = size(T, 2);
Xvar = zeros(numLVs, 1);
Yvar = zeros(numLVs, 1);

%Calculate the percent variance in X and Y
for i = 1:numLVs
    % Reconstruct X and Y using first i latent variables
    X_recon = T(:,1:i) * P(:,1:i)';
    X_resid = X_scaled - X_recon;
    Xvar(i) = 1 - sum(var(X_resid)) / sum(var(X_scaled));

    Y_pred_scaled = T(:,1:i) * Q(:,1:i)';
    Y_resid = Y_scaled - Y_pred_scaled;
    Yvar(i) = 1 - var(Y_resid) / var(Y_scaled);
end

Xvar = 100 * Xvar; %convert to percentage
Yvar = 100 * Yvar;

PLStable = [(1:numLVs)', ... %build PLS table
            [Xvar(1); diff(Xvar)], Xvar, ...
            [Yvar(1); diff(Yvar)], Yvar];

PLStable = array2table(PLStable, ...
    'VariableNames', {'LV', 'X_Var_%', 'X_Cum_%', 'Y_Var_%', 'Y_Cum_%'});



%% ---- Question 3 - Plot score plot LV1 vs LV2 for X
figure;
scatter(T(:,1),T(:,2));xlabel('PC1 T scores');ylabel('PC2 T Scores');



%% ---- Question 4 - Plot the LV1 weights of all variables
W = plsm.wts;  % extract the weights matrix
num_vars = 7;
num_timepoints = length(W) / num_vars;

W1 = W(:,1);   % extract weights for PC1 only
W_matrix = reshape(W1, 7, [])';

% Plot all weights
figure;
plot(W_matrix);
xlabel('time');
ylabel('LV1 weight');
ylim([-0.07 0.07]);
legend(var_names, 'Location', 'eastoutside');
title('PLS Weights for LV1 Over Time');
box on;

%% -- Weights alternative
for v=1:7
    figure;
    plot(W(v:7:end,1));
    xlabel('time');
    ylabel(['PC1 weights of ', var_names{v}]);
    ylim([-0.07 0.07])
end

figure; hold on;
for v=1:7
    plot(W(v:7:end,1), 'DisplayName', var_names{v});
end
xlabel('time');
ylabel('LV1 weight');
ylim([-0.07 0.07]);
legend(var_names, 'Location', 'eastoutside');
title('PLS Weights for LV1 Over Time');
box on;

%% ---- Question 5 - Regression Coefficients
B = plsm.reg;
num_vars = 7;
num_timepoints = length(B) / num_vars;

% Reshape B into a matrix: timepoints (rows) × variables (columns)
B_matrix = reshape(B, num_vars, num_timepoints)';  

% Plot all variables on one figure
figure;
plot(B_matrix);  % each column = one variable
xlabel('time');
ylabel('regression coefficient');
ylim([-0.01 0.01]);
legend(var_names, 'Location', 'eastoutside');
title('Regression Coefficients Over Time');
box on;


%% ---- Question 6 - Verify Linear Structure is Appropriate
% Plot the relationship between X scores and Y scores
figure;
scatter(T(:,1),U(:,1));xlabel('PC1 T scores');ylabel('PC1 U Scores');title('T vs U for LV1');
hold on;

% Add linear fit line
p = polyfit(T(:,1), U(:,1), 1); % fit a straight line
x_fit = linspace(min(T(:,1)), max(T(:,1)), 100);
y_fit = polyval(p, x_fit);
plot(x_fit, y_fit, 'r-', 'LineWidth', 1);
box on;

%% ---- Question 7 - Build Q vs T^2 monitoring chart with 95% CI
X_reconstructed = T * P';              % 
E = X_scaled - X_reconstructed;        % calc residuals

T2 = sum((T ./ std(T)).^2, 2);         % calc T2
SPE = sum(E.^2, 2);                    % Calc Q (SPE)

% calculate the limits with built in matlab code
T2_limit = tsqlim(93, 4, 0.95);
SPE_limit = residuallimit(E, 0.95);

figure;
scatter(T2, SPE, 50, 'filled');
xline(T2_limit, 'b--', 'T² 95% limit', 'LabelOrientation','horizontal', 'LabelVerticalAlignment','bottom');
yline(SPE_limit, 'r--', 'SPE 95% limit', 'LabelOrientation','horizontal', 'LabelVerticalAlignment','bottom');
% Add batch numbers to the plot
for i = 1:length(T2)
    text(T2(i), SPE(i), num2str(i), 'FontSize', 8, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
end
xlabel('T² (Hotelling)');
ylabel('Q (SPE)');
title('Q vs T² Monitoring Chart (Calibration Set)');
grid on;
box on;


%% ---- Question 8 - Build the matrices of residuals E and F
% Build E matrix - rebuilt like question 7
X_pred_scaled = T * P';              
E = X_scaled - X_pred_scaled; 

% Build F matrix
Y_pred = T * Q;
F = Y_scaled - Y_pred;

%% -- Plot residuals

% homoscedasticity
figure;
scatter(Y_pred, F);
xlabel('Predicted Y');
ylabel('Residuals');
title('Residuals vs. Predicted');
line(xlim, [0 0], 'Color', 'k');

% normality
figure;
qqplot(F);
title('Q-Q Plot of Residuals');

% independence
figure;
plot(F);
xlabel('Sample Index');
ylabel('Residual');
title('Residuals Across Samples');
line(xlim, [0 0], 'Color', 'k');


%% ---- Question 9 - Compute the mean relative error MRE
yhat_cal = modlpred(X2Dc,plsm,0);
rec = (Yc-yhat_cal)./Yc;
MREc=mean(abs(rec));


%% ---- Question 10 - Parity Plot in Calibration
R2c = 1-sum((Yc-yhat_cal).^2)/sum((Yc-repmat(mean(Yc),size(Yc))).^2);
figure;
scatter(Yc, yhat_cal, 50, 'filled');
hold on;
min_val = min([Yc; yhat_cal]);
max_val = max([Yc; yhat_cal]);
plot([min_val, max_val], [min_val, max_val], 'k--', 'LineWidth', 1);
xlabel('calibration measured y');
ylabel('predicted y');
box on;
xlim([min_val max_val]);
ylim([min_val max_val]);
text(min_val + 0.05*(max_val - min_val), ...
     max_val - 0.05*(max_val - min_val), ...
     sprintf('R^2 = %.3f', R2c), ...
     'FontSize', 12, 'FontWeight', 'bold', 'Color', 'blue');


%% ---- Question 11 - Preojection of validation data onto PLS model
ypredv = modlpred(X2Dv,plsm,0);
ev = Yv - ypredv;
rev = (Yv-ypredv)./Yv;
MREv = mean(abs(rev));


%% ---- Question 12 - Parity plot in validation
R2v = 1-sum((Yv-ypredv).^2)/sum((Yv-repmat(mean(Yv),size(Yv))).^2);
figure;
scatter(Yv, ypredv, 50, 'filled');
hold on;
min_valv = min([Yv; ypredv]);
max_valv = max([Yv; ypredv]);
plot([min_valv, max_valv], [min_valv, max_valv], 'k--', 'LineWidth', 1);
xlabel('validation measured y');
ylabel('predicted y');
box on;
xlim([min_valv max_valv]);
ylim([min_valv max_valv]);
text(min_valv + 0.05*(max_valv - min_valv), ...
     max_valv - 0.05*(max_valv - min_valv), ...
     sprintf('R^2 = %.3f', R2v), ...
     'FontSize', 12, 'FontWeight', 'bold', 'Color', 'blue');


%% ---- Question 13 - Projection of validation in Q vs T^2 chart
% scale the validation data using the mean and std of calibration data
Xcal_mean = mean(X2Dc);
Xcal_std = std(X2Dc);
Xval_scaled = (X2Dv - Xcal_mean) ./ Xcal_std;

T_val = Xval_scaled * P; % project onto PLS model to get scores

T2_val = sum((T_val ./ std(T)).^2, 2); % calc T^2

Xval_hat = T_val * P'; % reconstruct X using model
E_val = Xval_scaled - Xval_hat; % compute residuals
Q_val = sum(E_val.^2,2); % compute Q for validation

figure;
hold on;
scatter(T2, SPE, 40, 'b', 'filled');
scatter(T2_val, Q_val, 60, 'r', 'filled');
xline(T2_limit, 'b--', 'T² 95% limit', 'LabelOrientation','horizontal', 'LabelVerticalAlignment','bottom');
yline(SPE_limit, 'r--', 'SPE 95% limit', 'LabelOrientation','horizontal', 'LabelVerticalAlignment','bottom');
xlabel('T² (Hotelling)');
ylabel('Q (SPE)');
title('Q vs T² Monitoring Chart');
legend('Calibration', 'Validation', 'Location', 'best');
box on;


%% ---- Question 14 - Variable Contribution to Q and T^2 for Val Batch 2
E_val_i = E_val(2, :);  % select validation poiont outside of Q and T^2 range

% Reconstruct Q contribution by reshaping
nBatch = 93;
nVars = 7;             
nTime = 145;           
E_val_reshaped = reshape(E_val_i, [nVars, nTime]);

% Sum of squared errors for each variable (Q contribution)
Q_contrib = sum(E_val_reshaped.^2, 2);

% Compute mean and std of Q contributions from calibration set
E_reshaped_cal = reshape(E, [nBatch, nVars, nTime]);  
Q_contrib_cal = sum(E_reshaped_cal.^2, 3);  
Q_mean = mean(Q_contrib_cal, 1);
Q_std = std(Q_contrib_cal, 0, 1);

% Plot contribution plot for Q
figure;
hold on;
barColor = [0.2, 0, 0.4];
bar(Q_contrib, 'FaceColor', barColor);  % purple bars

plot(Q_mean + 2*Q_std, 'm--', 'LineWidth', 1.5);  % magenta upper
plot(Q_mean - 2*Q_std, 'm--', 'LineWidth', 1.5);  % magenta lower

xlabel('Variable');
ylabel('Q contribution');
title('Q Contribution Plot for Validation Batch');
box on;


% Calculate T^2 
W_reshaped = reshape(W, [7, 145, size(W,2)]);  
nVar = 7;
nTime = 145;
nCal = size(T, 1);
nLV = size(W, 2);

% Compute T² contributions for each calibration batch
T_contrib_cal = zeros(nCal, nVar);
for i = 1:nCal
    T_scaled_i = T(i,:) ./ std_T;  % [1 x LV]
    for v = 1:nVar
        W_v = squeeze(W_reshaped(v, :, :));  % [time x LV]
        proj = W_v * T_scaled_i';            % [time x 1]
        T_contrib_cal(i, v) = sum(proj.^2);  % scalar
    end
end

% Compute mean and ±2*std across calibration batches
T_mean = mean(T_contrib_cal, 1);   
T_std  = std(T_contrib_cal, 0, 1); 
upper_bound = T_mean + 2*T_std;
lower_bound = T_mean - 2*T_std;

% compute validation batch contribution
val_idx = 2;
T_scaled_val = T_val(val_idx,:) ./ std_T;
T_contrib_val = zeros(1, nVar);
for v = 1:nVar
    W_v = squeeze(W_reshaped(v, :, :));
    proj = W_v * T_scaled_val';
    T_contrib_val(v) = sum(proj.^2);
end

% Plot contribution plot for T^2
figure;
bar(T_contrib_val, 'FaceColor', barColor); hold on;
plot(1:nVar, upper_bound, '--m', 'Color', [1 0 1], 'LineWidth', 1.5);
plot(1:nVar, lower_bound, '--m', 'Color', [1 0 1], 'LineWidth', 1.5);
xlabel('Variable');
ylabel('T^2 Contribution');
title('T^2 Contribution by Variable (Validation Batch)');
box on;


%% --- Alternative Contribution Plots
tc = tconcalc(X2Dc, plsm);
qc = qconcalc(X2Dc, plsm);
tv = tconcalc(X2Dv, plsm);
qv = qconcalc(X2Dv, plsm);

for v=1:7
    figure
    bar(tv(:,v));
    hold on;
    
    mu = mean(tc(:,v));
    sigma = std(tc(:,v));
    plot([0, size(tv,1)], [mu - 1.96*sigma, mu - 1.96*sigma], '--m')
    plot([0, size(tv,1)], [mu + 1.96*sigma, mu + 1.96*sigma], '--m')

    box on;
    xlabel('sample');
    ylabel([var_names{v}, ' -T^2 contribution']);
    hold off
end

for v=1:7
    figure
    bar(qv(:,v));
    hold on;
    
    mu = mean(qc(:,v));
    sigma = std(qc(:,v));
    plot([0, size(qv,1)], [mu - 1.96*sigma, mu - 1.96*sigma], '--m')
    plot([0, size(qv,1)], [mu + 1.96*sigma, mu + 1.96*sigma], '--m')

    box on;
    xlabel('sample');
    ylabel([var_names{v}, ' -Q^2 contribution']);
    hold off
end



%% ---- Question 15 - Performance with LV+3

% new PLS model with 7 LVs
o.display='off';
o.plots='none';
o.preprocessing={'autoscale' 'autoscale'}; %autoscale the data
plsm2=pls(X2Dc,Yc,7,o); % Select 7LVs

T_new = plsm2.loads{1,1};  % X scores
P_new = plsm2.loads{2,1};  % X loadings
U_new = plsm2.loads{1,2};  % Y scores
Q_new = plsm2.loads{2,2};  % Y loadings
W_new = plsm2.wts;         % Weights 


% compute new MREc and MREv
yhat_cal_new = modlpred(X2Dc,plsm2,0);
rec_new = (Yc-yhat_cal_new)./Yc;
MREc_new = mean(abs(rec_new));

ypredv_new = modlpred(X2Dv,plsm2,0);
ev_new = Yv - ypredv_new;
rev_new = (Yv-ypredv_new)./Yv;
MREv_new = mean(abs(rev_new));

% compute new R^2 for calibration and validation
R2c_new = 1-sum((Yc-yhat_cal_new).^2)/sum((Yc-repmat(mean(Yc),size(Yc))).^2);
R2v_new = 1-sum((Yv-ypredv_new).^2)/sum((Yv-repmat(mean(Yv),size(Yv))).^2);

% Plot Q vs T^2 for calibration and validation
X_new = T_new * P_new';                
E_new = X_scaled - X_new;        % calc residuals

T2_new = sum((T_new ./ std(T_new)).^2, 2);         % calc T2
SPE_new = sum(E_new.^2, 2);                    % Calc Q (SPE)

% calculate the limits with built in matlab code
T2_limit_new = tsqlim(93, 4, 0.95);
SPE_limit_new = residuallimit(E_new, 0.95);

T_val_new = Xval_scaled * P_new; % project onto PLS model to get scores

T2_val_new = sum((T_val_new ./ std(T_new)).^2, 2); % calc T^2

Xval_hat_new = T_val_new * P_new'; % reconstruct X using model
E_val_new = Xval_scaled - Xval_hat_new; % compute residuals
Q_val_new = sum(E_val_new.^2,2); % compute Q for validation

figure;
hold on;
scatter(T2_new, SPE_new, 40, 'b', 'filled');
scatter(T2_val_new, Q_val_new, 60, 'r', 'filled');
xline(T2_limit_new, 'b--', 'T² 95% limit', 'LabelOrientation','horizontal', 'LabelVerticalAlignment','bottom');
yline(SPE_limit_new, 'r--', 'SPE 95% limit', 'LabelOrientation','horizontal', 'LabelVerticalAlignment','bottom');
xlabel('T² (Hotelling)');
ylabel('Q (SPE)');
title('Q vs T² Monitoring Chart');
legend('Calibration', 'Validation', 'Location', 'best');
box on;




