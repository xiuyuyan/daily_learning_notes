clc;clear all;close all;
Tp=10e-6;
f0=1e6;
B=2*f0;
fs=2*(f0+3*B);
t=-Tp/2:1/fs:Tp/2-1/fs;
Tr=1e-3;
tm=0:1/fs:Tr-1/fs;
n=round(Tp*fs);
N=round(Tr*fs);
x=zeros(1,N);
mu=B/Tp;
x(1:n)=exp(1j*2*pi*(f0*t+0.5*mu*t.^2));%发射信号
figure,plot(tm,real(x));
axis([tm(1) tm(end) 0 1.1])
xlabel('时间/s','FontSize',12)
ylabel('幅度','FontSize',12)
% 频谱
xf=fft(x)
Fs=-fs/2:fs/N:fs/2-fs/N;
figure,plot(Fs,fftshift(abs(xf)));
xlabel('频率(Hz)','FontSize',12)
ylabel('幅度','FontSize',12)
