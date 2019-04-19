%% Data Screening
clear all;
close all;
%% Initialization
dir_SHG_src = 'SHG_Batch';
dir_HE_src = 'HE_Batch';
dir_HE_tar = 'HE_Batch_sel';
dir_SHG_tar = 'SHG_Batch_sel';
rmdir(dir_HE_tar, 's')
rmdir(dir_SHG_tar, 's')
filename_SHG_sfx = '*.tif';
%% Create target dir
[status, msg, msgID] = mkdir(dir_HE_tar);
[status, msg, msgID] = mkdir(dir_SHG_tar);
%% Load all src images, compute entropy
file_info = dir([dir_SHG_src,'/',filename_SHG_sfx]);
for cnt_img = 1:size(file_info,1)
    filename_temp{cnt_img} = [dir_SHG_src,'/',file_info(cnt_img).name];
    SHG_temp = imread(filename_temp{cnt_img});
    entropy_SHG(cnt_img) = entropy(SHG_temp);
end
%% Sort entropy
[~,list_idx] = sort(entropy_SHG,'descend');
ratio_sel = 0.001;
Num_sel = round(ratio_sel*length(list_idx));
for cnt_sel = 1:Num_sel
    idx_img = list_idx(cnt_sel);
    filename_SHG_temp = [dir_SHG_src,'/',file_info(idx_img).name];
    filename_HE_temp = [dir_HE_src,'/',file_info(idx_img).name];
    movefile(filename_SHG_temp,dir_SHG_tar);
    movefile(filename_HE_temp,dir_HE_tar);
end
