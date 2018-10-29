%%
addpath(genpath('C:\Users\m_ana\Documents\Septimo\PD2\codigo\cuffofPD2\OpenPyMat\MPy\labstreaminglayer'))

%%
% instantiate the library
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


disp('Now receiving chunked data...');
a=[];
while true
    % get chunk from the inlet
 
    [chunk,stamps] = inlet.pull_chunk();
    a =[a,chunk];
    for s=1:length(stamps)
        % and display it
        fprintf('%.2f\t',chunk(:,s));
        fprintf('%.5f\n',stamps(s));
    
    end
%     plot(chunk (:,:,6))
 
end