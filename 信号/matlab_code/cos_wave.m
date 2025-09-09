clc;clear all;close all;
fs=10e6;% 采样频率
Tp=5e-3;% 脉冲宽度
t=-Tp/2:1/fs:Tp/2-1/fs; % 格式start : step : end
f0=4e3;% 载波频率
y=cos(2*pi*f0*t);% 生成余弦信号
figure,plot(t,y);axis([t(1) t(end) -1.2 1.2])
xlabel('时间/s','FontSize',12)
ylabel('幅度','FontSize',12)
% 频谱
yf=fft(y);
Fs=-fs/2:fs/length(yf):fs/2-fs/length(yf);
figure,plot(Fs,fftshift(abs(yf)));xlim([-20e3 20e3]);
xlabel('频率(Hz)','FontSize',12)
ylabel('幅度','FontSize',12)