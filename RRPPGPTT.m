%%
%leer los archivos
[d]=uigetdir();
folder=dir(d);
data=struct('num',[],'ppg',[],'ecg',[],'bp',[],'ptt',[],'RRint',[],'RRPeaks',[]);
data=repmat(data,1,1000);
k=1;
fs=125;

for i=1:length(folder)
    [~,~,ext]=fileparts(folder(i).name);
    
    if strcmp(ext,'.csv')
        nameFile=fullfile(d,folder(i).name);
        fid=fopen(nameFile);
        fileinfo=textscan(fid,'%f','Delimiter',',');
        fid=fclose(fid);
        fileinfo=[fileinfo{:}];
        maxLength=length(fileinfo)/3;
        data(k).num=k;
        data(k).ppg=fileinfo(1:maxLength);
        data(k).bp=fileinfo(maxLength+1:2*maxLength);
        data(k).ecg=fileinfo(2*maxLength+1:end);
        t=linspace(0,length(data(k).ecg)./fs,length(data(k).ecg));
        peak=findPeakEcg(data(k).ecg,t,1,0);
        data(k).RRint=t(peak);
        data(k).RRPeaks=data(k).ecg(peak);
        k=1+k;
      
        
    end
end

