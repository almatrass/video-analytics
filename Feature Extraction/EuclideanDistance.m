function dEuc = EuclideanDistance(sample1, sample2)

n=size(sample1,2);
total=0;

for i=1:n
    currentDistance=sample1(i)-sample2(i);
    currentDistance=currentDistance*currentDistance;
    total=total+currentDistance;
end

dEuc=sqrt(total);

end

