%% un archivo
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
%%
fs=125;
t=linspace(0,length(ecg)/125,length(ecg));
ax(1)=subplot(4,1,1);
plot(t,ecg)
ax(2)=subplot(4,1,2);
plot(t,ppt)
ax(3)=subplot(4,1,3);
plot(t,bbp)
maxs=diff(ecg);
RR=ecg;
RR(abs(maxs)~=0)=0;
ax(4)=subplot(4,1,4);
stem(t(abs(maxs)==0),RR(abs(maxs)==0))
linkaxes(ax,'x')
xlim([1 2]);
