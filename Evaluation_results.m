%% Data Screening
clear all;
close all;
%% Initialization
dir_SHG_tpl = ''; % load ground truth
dir_SHG_res = ''; % load result image

filename_SHG_sfx = '*.tif';
filename_save = 'Img_result.mat';

%% Load all ground truth and result images, compute PSNR and SSIM
file_info = dir([dir_SHG_tpl,'/',filename_SHG_sfx]);
for cnt_img = 1:size(file_info,1)
    % ensure that the filenames are the same in two dir
    filename_tpl{cnt_img} = [dir_SHG_tpl,'/',file_info(cnt_img).name];
    filename_res{cnt_img} = [dir_SHG_res,'/',file_info(cnt_img).name];
    
    SHG_tpl = imread(filename_tpl{cnt_img});
    SHG_res = imread(filename_res{cnt_img});
    
    [peaksnr_temp, snr_temp] = psnr(SHG_res, SHG_tpl);
    ssim_temp = ssim(SHG_res, SHG_tpl);
    
    PSNR_SHG(cnt_img) = peaksnr_temp;
    SNR_SHG(cnt_img) = snr_temp;
    SSIM_SHG(cnt_img) = ssim_temp;
end
%% save result
fprintf('PSNR = %d\n',mean(PSNR_SHG));
fprintf('SNR = %d\n',mean(SNR_SHG));
fprintf('SSIM = %d\n',mean(SSIM_SHG));
save(filename_save, 'PSNR_SHG','SNR_SHG','SSIM_SHG');

