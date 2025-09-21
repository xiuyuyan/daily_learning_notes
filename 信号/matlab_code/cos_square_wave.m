clc;clear all;close all;
Tp=100e-6;
fs=10e6;
t=-Tp/2:1/fs:Tp/2-1/fs;
Tr=1e-3;
tm=0:1/fs:Tr-1/fs;
n=Tp*fs;
N=Tr*fs;
E=10;
x=zeros(1,N);
f0=4e4;% 载波频率
% 信号生成方式1
x(1:n)=E*cos(2*pi*f0*t);
% 信号生成方式2
%{
x(1:n)=E;
x=x.*cos(2*pi*f0*tm);
%}
figure,plot(tm,x);
axis([tm(1) tm(end) 0 1.1*E]);
xlabel('时间/s','FontSize',12)
ylabel('幅度','FontSize',12)
% 频谱
xf=fft(x);
Fs=-fs/2:fs/N:fs/2-fs/N;
figure,plot(Fs,abs(fftshift(xf)));
xlim([-8e4 8e4]);
xlabel('频率(Hz)','FontSize',12)
ylabel('幅度','FontSize',12)