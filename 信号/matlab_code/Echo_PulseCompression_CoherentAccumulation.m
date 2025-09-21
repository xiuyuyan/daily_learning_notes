%----------回波+脉冲压缩+相参积累----------%
clc;clear all;close all;
% 相关参数
c=3e8;
f0=10e6;
Tp=10e-6;%脉冲宽度
Tr=1e-3;%脉冲重复周期
% Rmax=Tr*c/2;Rmin=Tp*c/2;
Rmin=10e3;Rmax=25e3; %虽然可以探测更大距离的目标，但我们一般只关注某一区域
B=2*f0;%带宽
fs=2*(f0+3*B);
mu=B/Tp;%调频率
Rwin=Rmax-Rmin;
Twin=2*Rwin/c;
Tstart=2*Rmin/c;%=Tp
Tend=2*Rmax/c;%=Tr

% 1.定义目标(单目标)
R0=12.5e3;
v=100;

% 1.1回波信号
Nwin=round(Twin*fs);
t=linspace(Tstart,Tend,Nwin);
tao=2*R0/c;
echo=exp(j*2*pi*(-f0*tao+0.5*mu*(t-tao-Tp/2).^2)).*(abs(t-tao-Tp/2)<Tp/2);

% 1.2多周期回波
M=2;
multi_R=repmat(R0,M,1)-v*Tr*(0:M-1)';
multi_tao=repmat(2*multi_R/c,1,length(t));
multi_t=repmat(t,M,1);
multi_echo=exp(j*2*pi*(-f0*multi_tao+0.5*mu*(multi_t-multi_tao-Tp/2).^2)).*(abs(multi_t-multi_tao-Tp/2)<Tp/2);
% 对比单周期和多周期，判断是否有错
% figure,plot(t,real(echo),'g-*');hold on,plot(t,real(multi_echo(1,:)),'b--');

% 2.定义目标（多目标）
R1=[12.1 12.101 18 20.2]'*1e3;
% R1=[12.1]'*1e3;

% 2.1回波信号
tao=repmat(R1*2/c,1,length(t));
tt=repmat(t,length(R1),1);
Echo=sum(exp(j*2*pi*(-f0*tao+0.5*mu*(tt-tao-Tp/2).^2)).*(abs((tt-tao-Tp/2))<Tp/2));
% Echo=(exp(j*2*pi*(-f0*tao+0.5*mu*(tt-tao-Tp/2).^2)).*(abs((tt-tao-Tp/2))<Tp/2));
figure(1),plot(t,real(Echo));title("目标回波");

% 3.脉冲压缩
% 3.1时域脉冲压缩
t0=linspace(0,Tp,Tp*fs);
ht=exp(-1j*pi*mu*t0.^2);
y=conv(Echo,ht);
R_win=linspace(Rmin,Rmax,length(t))-Tp/2*c/2;
rr=[R_win Rmax+1/fs*c/2*(1:Tp*fs-1)];
figure(2),plot(rr,abs(y));title("目标距离的脉冲压缩效果");
% 3.2频域脉冲压缩
hf=fft(ht,length(Echo));
yf=ifft(fftshift(fft(Echo).*hf));
hold on,plot(R_win,abs(yf),'r--');
legend("时域","频域");

