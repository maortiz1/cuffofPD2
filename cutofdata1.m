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
fs=125;
t=linspace(0,length(ecg)/125,length(ecg));
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
ec2=ecg.^2;
med=max(ec2);
des=std(ec2);
otecg=ecg;
otecg(ec2<(med-des)|ec2>(med+des))=0;

plot(t,ec2)

% xlim([1 2]);

% tr=maxs;
% media=mean(maxs(:));
%
% tr((media)>tr)=0;
% des=std(tr);
% med2=mean(media);
% tr((mean(tr)-des)>tr|(mean(tr)+des)<tr)=0;
% stem(t(2:end),tr)
linkaxes(ax,'x')

%%
ax2=subplot(4,1,1);
plot(t,ecg)
ax2(2)=subplot(4,1,2);
d=diff(ecg);
plot(t(2:end),d);
ax2(3)=subplot(4,1,3);
ecg2=d.^2;

windowSize = 5;

b = (1/windowSize)*ones(1,windowSize);
a=1;
eprom=filter(b,a,ecg2);
plot(t(2:end),ecg2)


ax2(4)=subplot(4,1,4);
peak=ecg2;

mas=max(peak);
des=std(peak);
peak((mas-des)<peak)=0;
tnew=t(2:end);
newpeak=diff(peak);
[ind]=find(peak~=0);
% x=diff(ind);
% y=find(x<6);
% y=y+1;
% ind(y)=[];
%  stem(tnew(ind),peak(ind)); 
plot(tnew(2:end),diff(peak))

linkaxes(ax2,'x')

%% find peaks
% 




