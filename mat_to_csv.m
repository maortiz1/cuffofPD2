
selpath = uigetdir
unzip(fullfile(selpath,'BloodPressureDataset.zip'))
numbers= [1:12];
for i = numbers
    name= sprintf('part_%i.mat',i);
    path_sample = fullfile(selpath, 'BloodPressureDataset',name)
    load(path_sample)
    for j = 1:numel(p)
        name_csv_file = sprintf('arc_%i.csv',j+1000*(i-1));
        path = fullfile('mat_to_csv', name_csv_file );
        csvwrite(path, p{1,j});
    end
end
