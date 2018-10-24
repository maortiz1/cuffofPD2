addpath(genpath('C:\Users\m_ana\Documents\Septimo\PD2\codigo\cuffofPD2\OpenBCI_MATLAB\Matlab-Python\labstreaminglayer'))
%% instantiate the library
disp('Loading the library...');
lib = lsl_loadlib();

% resolve a stream...
disp('Resolving an EEG stream...');
result = {};
while isempty(result)
    result = lsl_resolve_byprop(lib,'type','EEG'); end

% create a new inlet
disp('Opening an inlet...');
inlet = lsl_inlet(result{1});
[vec,ts] = inlet.pull_sample();
start = ts;
eeg_record = [];
while ts - start < 10
    [vec,ts] = inlet.pull_sample();
    eeg_record = [eeg_record;vec];
%     eeg_record = filtfilt([1.000 -2.026 2.148 -1.159 0.279],[0.028  0.053 0.071  0.053 0.028],eeg_record)
%     fprintf('%.2f\t',vec);
%     fprintf('%.5f\n',ts);
end
