% This function plots the bode diagram of Xw

% Authors(s): Yunjie Gu, Yitong Li

%% Notes:
%
% Xw: The complex value of the transfer function/matrix gain, i.e., Xw
% should NOT contain s. In addition, Xw should be a 3D matrix, i.e,
% Xw(M,N,W). [M,N] is the dimension of each Xw(:,:,W) at the frequency
% point fbd(W).
%
% fbd: Frequency vector in Hz. It should correpsond to Xw(:,:,W).

%%
function plot_c(Xw,fbd,varargin)

    [LineWidth,~]  = SimplusGT.LoadVar(1.5,'LineWidth',varargin);
    [LineStyle,~]  = SimplusGT.LoadVar('-','LineStyle',varargin);
    [Color,~]      = SimplusGT.LoadVar([],'Color',varargin);
    [PhaseOn,~]    = SimplusGT.LoadVar(1,'PhaseOn',varargin);
    [PhaseShift,~] = SimplusGT.LoadVar(0,'PhaseShift',varargin);
    [PlotOn,~]     = SimplusGT.LoadVar(1,'PlotOn',varargin);
    [ExportOn,~]   = SimplusGT.LoadVar(1,'ExportOn',varargin);
    [ExportFile,ExportFileFlag] = SimplusGT.LoadVar('bode_data.xlsx','ExportFile',varargin);
    [ExportBus,~]  = SimplusGT.LoadVar([],'ExportBus',varargin);

    [M,N,W] = size(Xw);

    if fbd(1)*fbd(W) < 0
        % seperate positive and negative frequency
        fbdn = fbd(1:W/2);
        fbdp = fbd(W/2+1:W);
        Xwn  = Xw(:,:,1:W/2);
        Xwp  = Xw(:,:,W/2+1:W);

        % anti wind up
        Arg_wn = angle(Xwn);
        Arg_wp = angle(Xwp);
        for m = 1:M
            for n = 1:N
                Arg_wn(m,n,:) = flip(Arg_wn(m,n,:));
                for k = 1:(W/2-1)
                    while (Arg_wn(m,n,k+1) - Arg_wn(m,n,k) > 1.5*pi)
                        Arg_wn(m,n,k+1) = Arg_wn(m,n,k+1) - 2*pi;
                    end
                    while (Arg_wn(m,n,k+1) - Arg_wn(m,n,k) < -1.5*pi)
                        Arg_wn(m,n,k+1) = Arg_wn(m,n,k+1) + 2*pi;
                    end
                end
                Arg_wn(m,n,:) = flip(Arg_wn(m,n,:));

                for k = 1:(W/2-1)
                    while (Arg_wp(m,n,k+1) - Arg_wp(m,n,k) > 1.5*pi)
                        Arg_wp(m,n,k+1) = Arg_wp(m,n,k+1) - 2*pi;
                    end
                    while (Arg_wp(m,n,k+1) - Arg_wp(m,n,k) < -1.5*pi)
                        Arg_wp(m,n,k+1) = Arg_wp(m,n,k+1) + 2*pi;
                    end
                end
            end
        end

        if PhaseOn == 0
            for m = 1:M
                for n = 1:N
                    if M*N > 1
                        figure();
                    end
                    subplot(1,2,1);
                    p(1)= loglog(fbdn,abs(squeeze(Xwn(m,n,:))));
                    grid on;  hold on;

                    subplot(1,2,2);
                    p(2)= loglog(fbdp,abs(squeeze(Xwp(m,n,:))));
                    grid on;  hold on;
                end
            end
        else
            for m = 1:M
                for n = 1:N
                    if PlotOn == 1
                        if M*N > 1
                            figure();
                        end
                        subplot(2,2,1);
                        p(1)= loglog(fbdn,abs(squeeze(Xwn(m,n,:))));
                        grid on;  hold on;

                        subplot(2,2,3);
                        p(2)= semilogx(fbdn,squeeze(Arg_wn(m,n,:)-PhaseShift)*180/pi);
                        grid on;  hold on;

                        subplot(2,2,2);
                        p(3)= loglog(fbdp,abs(squeeze(Xwp(m,n,:))));
                        grid on;  hold on;

                        subplot(2,2,4);
                        p(4)= semilogx(fbdp,squeeze(Arg_wp(m,n,:)+PhaseShift)*180/pi);
                        grid on;  hold on;
                    end

                end
            end

            if ExportOn == 1
                if ExportFileFlag == 0
                    Filename = 'bode_data.xlsx';
                else
                    Filename = ExportFile;
                end

                [fbdn_abs,idx_n] = sort(abs(fbdn(:)),'ascend');
                Xwn_export = Xwn(:,:,idx_n);
                Arg_wn_export = Arg_wn(:,:,idx_n);

                T_neg_mag_phase = localMagPhaseTable(fbdn_abs,Xwn_export,Arg_wn_export - PhaseShift,M,N);
                T_pos_mag_phase = localMagPhaseTable(fbdp(:),Xwp,Arg_wp + PhaseShift,M,N);
                T_neg_real_imag = localRealImagTable(fbdn_abs,Xwn_export,M,N);
                T_pos_real_imag = localRealImagTable(fbdp(:),Xwp,M,N);

                if isempty(ExportBus)
                    SheetNegMagPhase = 'Negative_MagPhase';
                    SheetPosMagPhase = 'Positive_MagPhase';
                    SheetNegRealImag = 'Negative_RealImag';
                    SheetPosRealImag = 'Positive_RealImag';
                else
                    SheetNegMagPhase = sprintf('Bus%d_Neg_MagPhase',ExportBus);
                    SheetPosMagPhase = sprintf('Bus%d_Pos_MagPhase',ExportBus);
                    SheetNegRealImag = sprintf('Bus%d_Neg_RealImag',ExportBus);
                    SheetPosRealImag = sprintf('Bus%d_Pos_RealImag',ExportBus);
                end
                SheetNegMagPhase = SheetNegMagPhase(1:min(length(SheetNegMagPhase),31));
                SheetPosMagPhase = SheetPosMagPhase(1:min(length(SheetPosMagPhase),31));
                SheetNegRealImag = SheetNegRealImag(1:min(length(SheetNegRealImag),31));
                SheetPosRealImag = SheetPosRealImag(1:min(length(SheetPosRealImag),31));

                writetable(T_neg_mag_phase, Filename, 'Sheet', SheetNegMagPhase);
                writetable(T_pos_mag_phase, Filename, 'Sheet', SheetPosMagPhase);
                writetable(T_neg_real_imag, Filename, 'Sheet', SheetNegRealImag);
                writetable(T_pos_real_imag, Filename, 'Sheet', SheetPosRealImag);
            end
        end
    else

        % anti wind up
        Arg_w = angle(Xw);
        for m = 1:M
            for n = 1:N
                for k = 1:(W-1)
                    while (Arg_w(m,n,k+1) - Arg_w(m,n,k) > 1.5*pi)
                        Arg_w(m,n,k+1) = Arg_w(m,n,k+1) - 2*pi;
                    end
                    while (Arg_w(m,n,k+1) - Arg_w(m,n,k) < -1.5*pi)
                        Arg_w(m,n,k+1) = Arg_w(m,n,k+1) + 2*pi;
                    end
                end
            end
        end

        if PhaseOn == 0
            for m = 1:M
                for n = 1:N
                    if M*N > 1
                        figure();
                    end
                    p = loglog(fbd,abs(squeeze(Xw(m,n,:))));
                    grid on;  hold on; 
                end
            end 
        else
            for m = 1:M
                for n = 1:N
                    if M*N > 1
                        figure();
                    end
                    subplot(2,1,1);
                    p(1)= loglog(fbd,abs(squeeze(Xw(m,n,:))));
                    grid on;  hold on;

                    subplot(2,1,2);
                    p(2)= semilogx(fbd,squeeze(Arg_w(m,n,:)+PhaseShift)*180/pi);
                    grid on;  hold on;  
                end
            end    
        end
    end
    
    try 
        p; %#ok<VUNUS>
        for h = 1:length(p)
            p(h).LineWidth = LineWidth;
            p(h).LineStyle = LineStyle;
            if ~isempty(Color)
                p(h).Color = Color;
            end
        end
    catch
    end

end

function T = localMagPhaseTable(fbd,Xw,Arg_w,M,N)

    Data = fbd(:);
    VarNames = {'Frequency_Hz'};
    for m = 1:M
        for n = 1:N
            EntryName = sprintf('%d%d',m,n);
            Mag = abs(squeeze(Xw(m,n,:)));
            Phase = squeeze(Arg_w(m,n,:))*180/pi;
            Data = [Data, Mag(:), Phase(:)]; %#ok<AGROW>
            VarNames = [VarNames, {['Magnitude_',EntryName], ['Phase_',EntryName,'_deg']}]; %#ok<AGROW>
        end
    end

    T = array2table(Data,'VariableNames',VarNames);
end

function T = localRealImagTable(fbd,Xw,M,N)

    Data = fbd(:);
    VarNames = {'Frequency_Hz'};
    for m = 1:M
        for n = 1:N
            EntryName = sprintf('%d%d',m,n);
            XwEntry = squeeze(Xw(m,n,:));
            Data = [Data, real(XwEntry(:)), imag(XwEntry(:))]; %#ok<AGROW>
            VarNames = [VarNames, {['Real_',EntryName], ['Imaginary_',EntryName]}]; %#ok<AGROW>
        end
    end

    T = array2table(Data,'VariableNames',VarNames);
end