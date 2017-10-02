% function[]  = writeFromPython2txt(healthCode,recordID,input,saveFile)
% to judege os and make path
clc
clear all

    strComputer = computer('arch')
    join = '\\';
    if strComputer(1) == 'w'%for windows
    readPath = 'D:\Parkinson Challenge\Parkinson Features\featuresFromCSV\tempData';
    savePath = 'D:\Parkinson Challenge\Parkinson Features\featuresFromCSV\finalFeatures';
    elseif strComputer(1) == 'm'%for mac
    join = '//';
    readPath = '../../../tempData';
    savePath = '../../../finalFeatures'; 
    end
    
    fileNameArr = dir(readPath);
    axesKey = 'xyz';
    for stepF = 3:length(fileNameArr)
        fileName = fileNameArr(stepF).name;
        sFileName = regexp(fileName,'_','split');
        healthCode = sFileName{1};
        recordID = sFileName{2};
        key = sFileName{3};
        inputPath = [readPath,join,fileName];
        input1 = importdata(inputPath);
        sizeI = size(input1);
        for stepA = 1:sizeI(2)
            input = input1(:,stepA)';
            saveFile = [savePath,join,key(1:end-4),axesKey(stepA),'.txt'];
            [energy, power, minValue, maxValue, medianValue, meanValue,...
            rmsValue, variance, stdValue,kurtosisValue, skewnessValue,...
            modeValue, trimMean, entropy, asyCoe, rangeValue, zeroCrossRate,...
            meanCrossRate, dfaValue, aveOn3,...
            aveOff3] = extractFeature(input);

            res = [energy, power, minValue, maxValue, medianValue, meanValue,...
            rmsValue, variance, stdValue,kurtosisValue, skewnessValue,...
            modeValue, trimMean, entropy, asyCoe, rangeValue, zeroCrossRate,...
            meanCrossRate, dfaValue, aveOn3,...
            aveOff3];

            fp = fopen(saveFile,'a');
            fprintf(fp,'%s\t%s\t',healthCode,recordID);
            for stepR = 1:length(res)
                fprintf(fp,'%f\t',res(stepR))
            end
            fprintf(fp,'\n');
            fclose(fp);
        end
    end
    
% end