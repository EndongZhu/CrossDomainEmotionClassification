clear ; close all; clc
%------------------------------------------------%
%fprintf('Initializing the data...\n');
data1 = importdata('trainingdata_1.mat') ;
data2 = importdata('trainingdata_2.mat') ;
data3 = importdata('trainingdata_3.mat') ;
train_num = 246 ;
test_num = 1000 ;
ratio = [1/64,1/32,1/16,1/8,1/4,1/2] ;
lam_set = [0.0001,0.001,0.01,0.1] ;
alpha_set = [0.5,0.7,1,1.3,2] ;

for ratio_id = 1:size(ratio,2)
    num_ratio = ratio(ratio_id) ;
    k_num = round(test_num*num_ratio) ;
    best_acc = 0 ;
    best_AP = -1 ;
    for lam_id = 1:size(lam_set,2)
        lam = lam_set(lam_id) ;
        for alpha_id = 1:size(alpha_set)
            alpha = alpha_set(alpha_id) ;

            X_train = data1(1:train_num,:) ;
            X_train_2 = data1(train_num+1:train_num+k_num,:) ;
            X_target = data1(train_num+k_num+1:train_num+test_num,:) ;


            Y_train = data2 ;
            Y_train_2 = data3(1:k_num,:) ;
            Y_target = data3(k_num+1:test_num,:) ;

            dim = size(Y_train,2) ;

            [~,Y_train] = max(Y_train,[],2) ;
            [~,Y_train_2] = max(Y_train_2,[],2) ;

            theta = zeros(dim ,size(X_train,2)) ;
            Beta = Gaussian(X_train , Y_train , X_train_2 , Y_train_2 , dim) ;

            %------------------------------------------------%
%            fprintf('training the data...\n');

            lambda = 20 ;
            [J_initial,grad] = CostFunction(theta , X_train , Y_train , X_train_2 , Y_train_2 , dim , Beta , alpha , 1 , lam) ;

            for i = 1 : 500
                [J,grad] = CostFunction(theta , X_train , Y_train , X_train_2 , Y_train_2 , dim , Beta , alpha , 1 , lam) ;
                %fprintf('iter#%d: %.5f\n',i,J);
                theta = theta - lambda * grad ;
            end

            %------------------------------------------------%
%            fprintf('making predictions...\n');

            temp = X_target * theta' ;
            likehood = zeros(size(X_target,1) , dim) ;

            for i = 1 : size(X_target,1)
                for j = 1 : dim
                    likehood(i,j) = exp(temp(i,j))/sum(exp(temp(i,:))) ;
                end
            end
            corr_ = zeros(size(likehood,1),size(Y_target,1));
            for i = 1:size(likehood,1)
                cc = corrcoef(likehood(i,:),Y_target(i,:));
                corr_(i,i) = cc(1,2);
            end
            corr_(isnan(corr_)) = 0;
            corr_num = sum(sum(corr_))/sum(sum(corr_~=0));

            [~,Y_target] = max(Y_target,[],2) ;
            [~,likehood] = max(likehood,[],2) ;
            num = mean(double(likehood == Y_target)) ;

            if(num > best_acc)
                best_acc = num ;
            end
            if(corr_num > best_AP)
                best_AP = corr_num ;
            end
        end
    end
    fprintf('num_ratio #%f:\n' , num_ratio);
    fprintf('the ACC is %f\n' , best_acc);
    fprintf('the AP is %f\n' , best_AP);
end
