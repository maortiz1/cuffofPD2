%%
%leer los archivos
[d]=uigetdir();
folder=dir(d);
data=struct('num',[],'ppg',[],'ecg',[],'bp',[],'ptt',[],'RRint',[],'')
for i=1:length(folder)
    [~,~,ext]=fileparts(folder(i).name);
    
    if strcmp(ext,'.csv')
        
        
    end
end