clc;clear all;close all;
% 参数定义
f0=0.2;f1=0.21;fs=0.5;N=50;M=500;deln=0.2;mu=0;
t=(0:N-1)/fs;snr=5;

% 定义信号
s=exp(j*2*pi*f0*t)+exp(j*2*pi*f1*t);
% figure,plot(t,s,'r-o');xlabel("时间/t");ylabel("幅值");
noise=1/sqrt(2)*(mvnrnd(mu,deln^2,N)+j*mvnrnd(mu,deln^2,N));
rou=10^(snr/10); % 信噪比：SNR=10*log10(rou)=10*log10(S/N)=>rou=S/N
x=rou*s+noise.';
% hold on,plot(t,x,'b-o');legend('clear signal','noise siginal');
% 频域信号（最初版本）
X=fft(x);Pxx=X.*conj(X);
t=0:fs/N:fs-fs/N;
% figure,plot(t,10*log10(Pxx/max(Pxx)),'b-');xlabel('时间/t');ylabel('频率/Hz');title("周期图");

% CG模型迭代 % ?不是假设X满足柯西分布吗？
X=fft(x,M);Pxx=X.*conj(X);
ff=0:fs/M:fs-fs/M;
Xx=fft(x);
figure,plot(t,10*log10(abs(Xx)/max(abs(Xx))));
hold on,plot(ff,10*log10(abs(X)/max(abs(X))),'m-.');

Jcg=inf;
F=exp(j*2*pi*(0:N-1)'*(0:M-1)/M)/M;
END=100;eps=1e-5;
for ii=1:END
    tmpJcg=Jcg;
    delx=cov(X);
    lamda=deln^2/delx;
    Q=diag(1+X.*conj(X)/2/delx);
    b=inv(lamda*eye(N)+F*Q*F')*x.';
    X=Q*F'*b;
    SX=sum(log(1+X.*conj(X)/2/delx));
    Jcg=SX+(x.'-F*X)'*(x.'-F*X)/2/deln^2;
%     hold on,plot(ff,10*log10(abs(X)/max(abs(X))),'r-.');
    figure,plot(ff,10*log10(abs(X)/max(abs(X))),'r-.');
    if abs(Jcg-tmpJcg)/(abs(Jcg)+abs(tmpJcg)) < eps*2
        disp(ii)
        break
    end
end
% F=exp(j*2*pi*(0:N-1)'*(0:M-1)/M)/M;
% delx=cov(X);
% lamda=deln^2/delx^2;
% Q=diag(1+Pxx/2/delx^2);
% b=inv(lamda*eye(N)+F*Q*F')*x.';
% tmpX=Q*F'*b;
% delx=cov(tmpX);
% hold on,plot(ff,10*log10(abs(tmpX)/max(abs(tmpX))),'r-.');
% SX=sum(log(1+tmpX.*conj(tmpX)/2/delx^2));
% Jcg=SX+(x.'-F*tmpX)'*(x.'-F*tmpX)/2/deln^2;
% END=100;eps=1e-5;
% for ii=1:END
%     tmpJcg=Jcg;
%     lamda=deln^2/delx^2;
%     Pxx=tmpX.*conj(tmpX);
%     Q=diag(1+Pxx/2/delx^2);
%     b=inv(lamda*eye(N)+F*Q*F')*x.';
%     tmpX=Q*F'*b;
%     delx=cov(tmpX);
%     SX=sum(log(1+tmpX.*conj(tmpX)/2/delx^2));
%     Jcg=SX+(x.'-F*tmpX)'*(x.'-F*tmpX)/2/deln^2;
%     hold on,plot(ff,10*log10(abs(tmpX)/max(abs(tmpX))),'r-.');
% %     figure,plot(ff,10*log10(abs(tmpX)/max(abs(tmpX))),'r-.');
% %     pause(0.5);
%     if abs(Jcg-tmpJcg)/(abs(Jcg)+abs(tmpJcg)) < eps*2
%         disp(ii)
%         break
%     end
% end
% 
% 



