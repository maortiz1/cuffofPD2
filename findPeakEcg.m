function [ind] = findPeakEcg( ecg ,facdes,height)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
ECGSQDEV=(diff(ecg).^2);
MAX=max(ECGSQDEV);
DES=std(ECGSQDEV);
ECGSQDEV((MAX-(facdes)*DES)<ECGSQDEV)=0;
[ind]=find(ECGSQDEV>height);
x=diff(ind);
y=find(x<6);
y=y+1;
ind(y)=[];
end

