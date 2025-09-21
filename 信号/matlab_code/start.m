clc;clear all;close all;
fs=10e6;
Tp=5e-3;%脉冲宽度
t=-Tp/2:1/fs:Tp/2-1/fs;
f0=4e3;%载频频率
y=cos(2*pi*f0*t);
figure(1),plot(t,y,'b-'),axis([t(1) t(end) -1.2 1.2]);
xlabel("t/s"),ylabel("幅度");
%--傅里叶变换--%
M=1024;%将时域信号转变为M个频域信号
% yf=fft(y,M);% 指离散傅里叶变换，若省略M则表示yf与y的点数一致
% Fs=-fs/2:fs/M:fs/2-1/M;%频率轴
yf=fft(y);
Fs=-fs/2:fs/length(yf):fs/2-fs/length(yf);
figure(2),plot(Fs,fftshift(abs(yf)),'r-');
axis([-6000 6000 0 max(abs(yf))]);
xlabel("频率/Hz"),ylabel("幅度");

figure(3),subplot(1,2,1);
plot(t,y,'b-'),axis([t(1) t(end) -1.2 1.2]);
xlabel("t/s"),ylabel("幅度");
subplot(1,2,2);
plot(Fs,fftshift(abs(yf)),'r-');
axis([-6000 6000 0 max(abs(yf))]);
xlabel("频率/Hz"),ylabel("幅度");