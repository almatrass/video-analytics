function [Lut] = contrast_LS_LUT(m,c)

Lut=zeros(1,256);
for i=1:256
    if i<(-c/m)
        Lut(i)=0;
    else 
        if i>(255-c)/m
            Lut(i)=255;
        else
            Lut(i)=(m*i)+c;
        end
    end        
end
Lut=uint8(Lut);
end