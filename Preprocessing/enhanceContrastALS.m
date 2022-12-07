function Iout = enhanceContrastALS(Iin)

IHist=histogram(Iin,'BinLimits',[0 256],'BinWidth',1);
IHistValues=IHist.Values;

range=find(IHistValues > 9);
I1=range(1);
I2=range(end);
m=255/(I2-I1);

c=-(m*I1);

Lut = contrast_LS_LUT(m,c);
Iout = intlut(Iin,Lut);

end

