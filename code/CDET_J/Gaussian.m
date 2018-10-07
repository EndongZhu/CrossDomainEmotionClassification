function [Beta] = Gaussian(X_s , Y_s , X_t , Y_t , dim)
    size1 = size(X_s , 1);
    size2 = size(X_t , 1);
    
    It = ones(size1 , size2) ;
    Is = ones(size1 , size1) ;
    Beta = ones(size1 , 1) ;
    
    for i = 1 : size1
        for j = 1 : size2
            if Y_s(i) == Y_t(j)
                It(i,j) = 1 ;
            else
                It(i,j) = 0 ;
            end
        end
    end
    
    for i = 1 : size1
        for j = 1 : size1
            if Y_s(i) == Y_s(j)
                Is(i,j) = 1 ;
            else
                Is(i,j) = 0 ;
            end
        end
    end
    
    sigma = 10 ;
    for i = 1 : size1
        var1 = 0 ;
        for j = 1 : size2
            xi = X_s(i , :)' ;
            xj = X_t(j , :)' ;
            temp = exp(-norm(xi-xj)/(sigma*sigma)) ;
            var1 = var1 + It(i,j)*temp ;
        end
        if sum(It(i,:)) ~= 0
            var1 = var1/sum(It(i,:)) ;
            
        else
            var1 = 0 ;
        end
        var2 = 0 ;
        for j = 1 : size1
            xi = X_s(i , :)' ;
            xj = X_s(j , :)' ;
            temp = exp(-norm(xi-xj)/(sigma*sigma)) ;
            var2 = var2 + Is(i,j)*temp - 1 ;
        end
        var2 = var2/sum(Is(i,:)-ones(1,size1)) ;
        
        pr1 = (var1)/(var2) ;
        pr2 = (sum(It(i,:))/size2+0.001) / (sum(Is(i,:))/size1+0.001) ;
        Beta(i) = pr1*pr2 ;
    end



end
