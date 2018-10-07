function [J,grad] = CostFunction( theta , X , Y , X_2 , Y_2 , dim , Beta , lambda_t , lambda_s , lambda_r)
%COSTFUNCTION Summary of this function goes here
%   Detailed explanation goes here

    grad = zeros(size(theta));
    m = size(Y , 1) ;
    m_2 = size(Y_2 , 1) ;
    
    temp = exp(X * theta') ;
    temp_2 = exp(X_2 * theta') ;
    neg_likehood = zeros(size(X,1) , 1) ;
    neg_likehood_2 = zeros(size(X_2,1) , 1) ;
    
    for i = 1 : m
        for j = 1 : dim
            if Y(i) == j
                neg_likehood(i) =  Beta(i)*log(temp(i,j) / sum(temp(i,:))) ;
            end
        end
    end
    
    for i = 1 : m_2
        for j = 1 : dim
            if Y_2(i) == j
                neg_likehood_2(i) =  log(temp_2(i,j) / sum(temp_2(i,:))) ;
            end
        end
    end
    
    hh = sum(theta.*theta) ;
    J = lambda_t*sum(neg_likehood_2)/(-m_2)+ lambda_s*sum(neg_likehood)/(-m)  + lambda_r*sum(hh)/(2*(m+m_2)) ;
    
    for i = 1 : dim
        var = zeros(1,size(grad,2)) ;
        for j = 1 : m
            if Y(j) == i
                var = var + Beta(j)*X(j,:)*(1-(temp(j,i) / sum(temp(j,:)))) ;
            else
                var = var + Beta(j)*X(j,:)*(-temp(j,i) / sum(temp(j,:))) ;
            end
        end
        
        var_2 = zeros(1,size(grad,2)) ;
        for j = 1 : m_2
            if Y_2(j) == i
                var_2 = var_2 + X_2(j,:)*(1 - (temp_2(j,i) / sum(temp_2(j,:)))) ;
            else
                var_2 = var_2 + X_2(j,:)*(-temp_2(j,i) / sum(temp_2(j,:))) ;
            end
        end
        
        grad(i,:) = var_2 * (lambda_t/-m_2) + var * (lambda_s/-m) + lambda_r*theta(i,:)/(m+m_2) ;
    end

end

