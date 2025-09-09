clc;clear all;close all;
%----------雷达参数设置----------%
c=3e8;
f0=10e6;
Tp=10e-6;
B=2*f0;
Tr=1e-3;
% Rmin=Tp*c/2;Rmax=Tr*c/2;
Rmin=10e3;Rmax=25e3;
v=100;% 目标运动速度
fs=2*(f0+3*B);%采样频率
mu=B/Tp;
Rwin=Rmax-Rmin;%可探测窗口（距离）
Twin=2*Rwin/c;%可探测窗口（时间）
Tstart=2*Rmin/c;Tend=2*Rmax/c;
Nwin=round(Twin*fs);
t=linspace(Tstart,Tend,Nwin);
%----------单目标回波----------%
R0=12.5e3;
tao=2*R0/c*ones(1,length(t));
Echo=exp(j*2*pi*(-f0*tao+0.5*mu*(t-tao-Tp/2).^2)).*(abs(t-tao-Tp/2)<Tp/2);
% figure,plot(t,real(Echo))
% title("单周期单目标回波")

%----------多周期单目标回波----------%
M=2;
R1=repmat(R0,M,1)-v*Tr*(0:M-1)';
tao1=repmat(2*R1/c,1,length(t));
td1=repmat(t,M,1)-tao1-Tp/2;
Echo1=exp(j*2*pi*(-f0*tao1+0.5*mu*td1.^2)).*(abs(td1)<Tp/2);
figure(1),plot(t,real(Echo),'g-*');
hold on,plot(t,real(Echo1(1,:)),'b--');
hold on,plot(t,real(Echo1(2,:)),'r--');
axis([t(1) t(end) -1.5 1.5]);
legend('单周期回波','多周期回波(1)','多周期回波(2)')
title("多周期单目标回波")
%----------多目标回波----------%
R1=[12.1 14.8 18 20.2]'*1e3;
tao=repmat(R1*2/c,1,length(t));
tt=repmat(t,length(R1),1);
% Echo2=sum(exp(j*2*pi*(-f0*tao+0.5*mu*(tt-tao-Tp/2).^2)).*(abs(tt-tao-Tp/2)<Tp/2));
Echo2=exp(j*2*pi*(-f0*tao+0.5*mu*(tt-tao-Tp/2).^2)).*(abs(tt-tao-Tp/2)<Tp/2);
figure(2),plot(tt,real(Echo2(1,:)),'g--');
% hold on,plot(tt,real(Echo2(2,:)),'r--');
hold on,plot(tt,real(Echo2(3,:)),'b--');
% hold on,plot(tt,real(Echo2(4,:)),'y--');
% figure(2),plot(tt,real(Echo2),'g--');