function [ind] = findPeakEcg( ecg ,t,facdes,inv)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
ECGSQDEV=(diff(ecg).^2);
MAX=max(ECGSQDEV);
DES=std(ECGSQDEV);
ECGSQDEV((MAX-(facdes)*DES)<ECGSQDEV)=0;
if inv>0
MAX=max(ECGSQDEV);
DES=std(ECGSQDEV);
[ind]=find(ECGSQDEV>(MAX-2*DES));
else
    [ind]=find(ECGSQDEV>0.01);
end
x=diff(ind);
y=find(x<6);
y=y+1;
ind(y)=[];
ind=ind+1;
tdif=diff(t(ind));
ind2=find(tdif<0.5);
ind(ind2+1)=[];

end

