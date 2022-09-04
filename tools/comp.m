
clear; warning('off','all');
imageshow = 0;
minval=3;

fignum          = 500;
new_siga_2n_2   = 0.1; % up => gamma down,     down => gamma up
searchRange     = 1;
searchStep      = 0.01;

if imageshow==1
    searchRange=1;
end
flag_savepng    = 0;
sigRate = new_siga_2n_2-(floor(searchRange/2)*searchStep):searchStep:new_siga_2n_2+(ceil(searchRange/2)*searchStep)-searchStep+1e-15;
% sigRate = ones(1,searchRange)*new_siga_2n_2;
str_res=sprintf('adv%.2f', new_siga_2n_2);

myDirs = strings(1,100); oo = 1;
resc(oo)=1; x4(oo)=0; pr(oo)=imageshow; myDirs(oo) = '../results_real_lr_x2_w0.01_dvar777.0_g5_d5_v0.8/real_x2_000'; oo=oo+1;
% resc(oo)=0; x4(oo)=0; pr(oo)=imageshow; myDirs(oo) = '../results_real0.07/real_x2_777'; oo=oo+1;



for ooo=1:(oo-1)
    fprintf(1,'%s\n', myDirs(ooo));
    if not(exist(fullfile(myDirs(ooo), str_res),'dir'))
        mkdir(fullfile(myDirs(ooo), str_res))
    end
    
    if x4(ooo)==0
        s=2.0;
        gSize = 17;
        gtDir = '../../data/DIV2KRK/gt_k_x2';
    else
        s=4.0;
        gSize = 31;
        gtDir = '../../data/DIV2KRK/gt_k_x4';
    end
    gtFiles = dir(fullfile(gtDir,'*.mat'));
    gtFiles = natsortfiles({gtFiles.name});

    sum_L2_1x1  = zeros(1,searchRange);
    sum_L2_3x3  = zeros(1,searchRange);
    sum_L2_5x5  = zeros(1,searchRange);
    
    sum_L2_m1x1  = zeros(1,searchRange);
    sum_L2_m3x3  = zeros(1,searchRange);
    sum_L2_m5x5  = zeros(1,searchRange);
    
    sum_L2      = zeros(1,searchRange);
    sum_difMax  = zeros(1,searchRange);
    
    sumvar_x    = zeros(1,searchRange);
    sumvar_y    = zeros(1,searchRange);
    sumsf_x     = zeros(1,searchRange);
    sumsf_y     = zeros(1,searchRange);
    sumvar      = zeros(1,searchRange);
    sumsf       = zeros(1,searchRange);
    kkk = 0;
    for k = 1:100
        myFileName = fullfile(myDirs(ooo), ['nonc_im_' int2str(k) '.mat']);
        if isfile(myFileName)
            data = load(myFileName, 'Kernel');
            Kernel = centerization(data.Kernel);
            data = load (fullfile(gtDir, cell2mat(gtFiles(k))), 'Kernel');
            gtKernel = gt_pad(data.Kernel, gSize);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            rotrot = get_rotation_angle(Kernel);
            Kernel = imrotate(Kernel,rotrot, 'bilinear', 'crop');
            Kernel = centerization(Kernel);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [var_x, var_y] = get_var(Kernel);
            
            
            for idx = 1:searchRange
                if minval==0
                    ax_rate  = min(0.0, (sigRate(idx)/var_x));
                    ay_rate  = min(0.0, (sigRate(idx)/var_y));
                elseif minval==1
                    ax_rate  = min(sigRate(idx), (sigRate(idx)/var_x));
                    ay_rate  = min(sigRate(idx), (sigRate(idx)/var_y));
                elseif minval==2
                    ax_rate  = min((s^2)/3-1, (sigRate(idx)/var_x));
                    ay_rate  = min((s^2)/3-1, (sigRate(idx)/var_y));
                elseif minval==3
                    ax_rate  = min(3/s^2, (sigRate(idx)/var_x));
                    ay_rate  = min(3/s^2, (sigRate(idx)/var_y));
                elseif minval==4
                    ax_rate  = min(1.0, (sigRate(idx)/var_x));
                    ay_rate  = min(1.0, (sigRate(idx)/var_y));
                end
                new_sf_x = s/sqrt(3) * sqrt(1-(3/(s^2)) * ax_rate);
                new_sf_y = s/sqrt(3) * sqrt(1-(3/(s^2)) * ay_rate);

                ker = custom_resize(Kernel, [new_sf_x new_sf_y]);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                ker = set_to_outsize(ker, gSize);
                ker = centerization(ker);
                ker = imrotate(ker, -rotrot, 'bilinear', 'crop');
                ker = centerization(ker);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                centr  = (gSize+1)/2;
                difMax = max(max(ker)) - max(max(gtKernel));
                l2_1x1 = sum((ker(centr, centr) - gtKernel(centr, centr)).^2,'all');
                l2_3x3 = sum((ker(centr-1:centr+1, centr-1:centr+1) - gtKernel(centr-1:centr+1, centr-1:centr+1)).^2,'all');
                l2_5x5 = sum((ker(centr-2:centr+2, centr-2:centr+2) - gtKernel(centr-2:centr+2, centr-2:centr+2)).^2,'all');
                
                l2_m1x1 = ker - gtKernel;
                l2_m1x1(centr, centr)=0;
                l2_m1x1 = sum(l2_m1x1.^2,'all');
                l2_m3x3 = ker - gtKernel;
                l2_m3x3(centr-1:centr+1, centr-1:centr+1)=0;
                l2_m3x3 = sum(l2_m3x3.^2,'all');
                l2_m5x5 = ker - gtKernel;
                l2_m5x5(centr-2:centr+2, centr-2:centr+2)=0;
                l2_m5x5 = sum(l2_m5x5.^2,'all');
                
                l2_all = sum((ker - gtKernel).^2,'all');
                
                sumvar_x(idx) = sumvar_x(idx) + var_x;
                sumvar_y(idx) = sumvar_y(idx) + var_y;
                sumvar(idx)   = sumvar(idx)   + (var_x+var_y)/2.0;
                
                sumsf_x(idx)  = sumsf_x(idx) + new_sf_x;
                sumsf_y(idx)  = sumsf_y(idx) + new_sf_y;
                sumsf(idx)    = sumsf(idx)   + (new_sf_x+new_sf_y)/2.0;

                sum_L2_1x1(idx)  = sum_L2_1x1(idx)    + l2_1x1;
                sum_L2_3x3(idx)  = sum_L2_3x3(idx)    + l2_3x3;
                sum_L2_5x5(idx)  = sum_L2_5x5(idx)    + l2_5x5;
                
                sum_L2_m1x1(idx)  = sum_L2_m1x1(idx)    + l2_m1x1;
                sum_L2_m3x3(idx)  = sum_L2_m3x3(idx)    + l2_m3x3;
                sum_L2_m5x5(idx)  = sum_L2_m5x5(idx)    + l2_m5x5;
                
                sum_L2(idx)      = sum_L2(idx)        + l2_all;
                sum_difMax(idx)  = sum_difMax(idx)    + difMax;
            end
            
            Kernel = ker;
            % rescale & save
            if resc(ooo)==1
                save_kernel(Kernel, x4(ooo), myDirs(ooo), str_res, k);
            end
            if flag_savepng==1
                pngname = ['ker_' int2str(k) '.png'];
                save_png(Kernel, pngname);
            end
            if (k<=98) && (pr(ooo)==1) && (flag_savepng==0)
                newStr = [int2str(k) ', ' num2str(l2_all, '%3.4f')];
                fig=figure(fignum); subplot(7,98/7,k); imagesc(Kernel); title(newStr);
            end
            kkk = kkk+1;
        end
    end
    if (pr(ooo)==1) && (flag_savepng==0)
        fig=figure(fignum); 
        ststr = sprintf('%s %s %s %f', pwd, str_res, myDirs(ooo), sum_L2(idx)/kkk);
        newStr = regexprep(ststr,'_','-');
        sgtitle(newStr);
        
        pngname = pwd;
        if x4(ooo)==0
            pngname = fullfile('O:\ys\sr\results_image', [pngname(10:end-6) '.png']);
        else
            pngname = fullfile('O:\ys\sr\results_image', ['x4_' pngname(10:end-6) '.png']);
        end
        saveas(fig,pngname,'png');
    end
    fignum = fignum + 10;
    for idx = 1:searchRange
        fprintf(1, '[%d, %.2f], %3.6f, %3.6f, %3.6f, %3.6f, %3.6f, %3.6f, %3.6f, %3.6f\n', kkk, sigRate(idx), sum_L2(idx)/kkk, sum_difMax(idx)/kkk, sum_L2_1x1(idx)/kkk, sum_L2_3x3(idx)/kkk, sum_L2_5x5(idx)/kkk, sum_L2_m1x1(idx)/kkk, sum_L2_m3x3(idx)/kkk, sum_L2_m5x5(idx)/kkk);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rotrot = get_rotation_angle(in_ker)
    if (size(in_ker,1)~=size(in_ker,2)) || (mod(size(in_ker,2), 2) == 0)
        error;
    end
    kersize = size(in_ker,2);      
    [X,Y] = meshgrid(1:kersize);
    X = X-(kersize+1)/2;
    Y = Y-(kersize+1)/2;
    sample_x=[];
    sample_y=[];
    for thres = 0.1:0.1:0.9
        mask = (in_ker>max(max(in_ker))*0.1);
        for idxX=1:kersize
            for idxY=1:kersize
                if mask(idxX, idxY)==1
                    sample_x=[sample_x; X(idxX, idxY)];
                    sample_y=[sample_y; Y(idxX, idxY)];
                end
            end
        end
    end
    data = [sample_x sample_y];

    covariance = cov(data);
    [eigenvec, eigenval ] = eig(covariance);

    [largest_eigenvec_ind_c, r] = find(eigenval == max(max(eigenval)));
    largest_eigenvec = eigenvec(:, largest_eigenvec_ind_c);
%     largest_eigenval = max(max(eigenval));
% 
%     if(largest_eigenvec_ind_c == 1)
%         smallest_eigenval = max(eigenval(:,2));
%         smallest_eigenvec = eigenvec(:,2);
%     else
%         smallest_eigenval = max(eigenval(:,1));
%         smallest_eigenvec = eigenvec(1,:);
%     end

    angle = atan2(largest_eigenvec(2), largest_eigenvec(1));
    if(angle < 0)
        angle = angle + 2*pi;
    end
    rotrot = angle/pi*180;
end

function in_ker = centerization(in_ker)
    if (size(in_ker,1)~=size(in_ker,2)) || (mod(size(in_ker,2), 2) == 0)
        error;
    end
    kersize = size(in_ker,2);
    in_ker  = in_ker.*(in_ker>0);
    props   = regionprops(true(size(in_ker)), in_ker, 'WeightedCentroid');
    in_ker  = imtranslate(in_ker,(kersize+1)/2-props.WeightedCentroid);
    in_ker  = in_ker/sum(in_ker,'all');
end

function in_ker = gt_pad(in_ker, outSize)
    if (size(in_ker,1)~=size(in_ker,2)) || (mod(size(in_ker,2), 2) == 0)
        error;
    end
    pd = (outSize-size(in_ker, 2))/2;
    in_ker = padarray(in_ker, [pd, pd], 0);
    
    kersize = size(in_ker,2);
    props   = regionprops(true(size(in_ker)), in_ker, 'WeightedCentroid');
    in_ker  = imtranslate(in_ker,(kersize+1)/2-props.WeightedCentroid);
end

function [var_x, var_y] = get_var(in_ker)
    if (size(in_ker,1)~=size(in_ker,2)) || (mod(size(in_ker,2), 2) == 0)
        error;
    end
    kersize = size(in_ker,2);
    cnt = (kersize+1)/2;
    xxx = sum(in_ker,2);
    yyy = sum(in_ker,1);
    xxx1=log(xxx(cnt)/xxx(cnt+1));
    xxx2=log(xxx(cnt+1)/xxx(cnt+2));
    yyy1=log(yyy(cnt)/yyy(cnt+1));
    yyy2=log(yyy(cnt+1)/yyy(cnt+2));

    % 3번째 자리의 값이 너무 작으면 xxx2,yyy2가 매우 커지고 var이 너무 작아지게 된다.
    % 3번째자리 0일땐 오류
    if xxx2 > (xxx1*3)
        var_x = 1/(xxx1*4); % 2*sig^2 / 4*n^2
    else
        var_x = 1/(xxx1+xxx2); % 2*sig^2 / 4*n^2
    end
    if yyy2 > (yyy1*3)
        var_y = 1/(yyy1*4); % 2*sig^2 / 4*n^2
    else
        var_y = 1/(yyy1+yyy2); % 2*sig^2 / 4*n^2
    end
end


function out_ker = set_to_outsize(in_ker, outSize)
    out_ker = zeros(outSize*2-1,outSize*2-1); % 17 => 33, 31 => 61
    out_ker(1:size(in_ker,1), 1:size(in_ker,2))=in_ker;
    
    props   = regionprops(true(size(out_ker)), out_ker, 'WeightedCentroid');
    out_ker = imtranslate(out_ker,(size(out_ker,1)+1)/2-props.WeightedCentroid);
    cnt = (size(out_ker,1)+1)/2;
    pd = (outSize-1)/2;
    
    out_ker = out_ker(cnt-pd:cnt+pd, cnt-pd:cnt+pd);
end

function save_png(Kernel, name)
    figure(54321);
    imagesc(Kernel);
    set(gca,'XTick',[]) % Remove the ticks in the x axis!
    set(gca,'YTick',[]) % Remove the ticks in the y axis
    set(gca,'Position',[0 0 1 1]) % Make the axes occupy the hole figure
    saveas(gcf,name,'png');
end

function save_kernel(Kernel, x4, myD, str_save, idx)
    if x4==0
        fileN = ['nonc_kernel_' int2str(idx) '_x2.mat'];
    else
        fileN = ['nonc_kernel_' int2str(idx) '_x4.mat'];
    end
    saveName = fullfile(myD, str_save, fileN);
    save(saveName, 'Kernel');
end
