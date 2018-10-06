clc
clear all
%toamr un archivo
[f,d]=uigetfile(cd);
ruta=fullfile(d,f);
fid=fopen(ruta);
data=textscan(fid,'%f','Delimiter',',');
maxd=length([data{:}])/3;
data=[data{:}];
ppt=data(1:maxd);
bbp=data(maxd+1:2*maxd);
ecg=data(2*maxd+1:end);
fid=fclose(fid);
fs=125;
t=linspace(0,length(ecg)/fs,length(ecg));
%%
[ind]=findPeakEcg(ppt,t,1,0);
%%
figure
plot(t,ecg,'b',t(ind),ecg(ind),'r*')%,t(ind),ecg(ind),'*')
% xlim([1 3])
%%
