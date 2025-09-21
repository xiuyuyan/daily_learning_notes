clc;clear all;close all;
Tp=100e-6;%脉冲宽度
fs=10e6;
%t=-Tp/2:1/fs:Tp/2-1/fs;
Tr=1e-3;% 脉冲重复周期
tm=0:1/fs:Tr-1/fs;
n=Tp*fs;
N=Tr*fs;
E=10;
x=zeros(1,N);
x(1:n)=E;
figure,plot(tm,x);
axis([tm(1) tm(end) 0 1.1*E])
xlabel('时间/s','FontSize',12)
ylabel('幅度','FontSize',12)
% 频谱
xf=fft(x);
Fs=-fs/2:fs/N:fs/2-fs/N;
figure,plot(Fs,fftshift(abs(xf)));xlim([-20e4 20e4])
xlabel('频率(Hz)','FontSize',12)
ylabel('幅度','FontSize',12)