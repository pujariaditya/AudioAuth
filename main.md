%% V1.4b
%% 2015/08/26
%% by Michael Shell
%% see http://www.michaelshell.org/
%% for current contact information.
%%
%% This is a skeleton file demonstrating the use of IEEEtran.cls
%% (requires IEEEtran.cls version 1.8b or later) with an IEEE
%% journal paper.
%%
%% Support sites:
%% http://www.michaelshell.org/tex/ieeetran/
%% http://www.ctan.org/pkg/ieeetran
%% and
%% http://www.ieee.org/

%%*************************************************************************
%% Legal Notice:
%% This code is offered as-is without any warranty either expressed or
%% implied; without even the implied warranty of MERCHANTABILITY or
%% FITNESS FOR A PARTICULAR PURPOSE! 
%% User assumes all risk.
%% In no event shall the IEEE or any contributor to this code be liable for
%% any damages or losses, including, but not limited to, incidental,
%% consequential, or any other damages, resulting from the use or misuse
%% of any information contained here.
%%
%% All comments are the opinions of their respective authors and are not
%% necessarily endorsed by the IEEE.
%%
%% This work is distributed under the LaTeX Project Public License (LPPL)
%% ( http://www.latex-project.org/ ) version 1.3, and may be freely used,
%% distributed and modified. A copy of the LPPL, version 1.3, is included
%% in the base LaTeX documentation of all distributions of LaTeX released
%% 2003/12/01 or later.
%% Retain all contribution notices and credits.
%% ** Modified files should be clearly indicated as such, including  **
%% ** renaming them and changing author support contact information. **
%%*************************************************************************

% *** Authors should verify (and, if needed, correct) their LaTeX system  ***
% *** with the testflow diagnostic prior to trusting their LaTeX platform ***
% *** with production work. The IEEE's font choices and paper sizes can   ***
% *** trigger bugs that do not appear when using other class files.       ***                          ***
% The testflow support page is at:
% http://www.michaelshell.org/tex/testflow/

\documentclass[journal]{IEEEtran}
\usepackage{comment}
%
% If IEEEtran.cls has not been installed into the LaTeX system files,
% manually specify the path to it like:
% \documentclass[journal]{../sty/IEEEtran}

% Some very useful LaTeX packages include:
% (uncomment the ones you want to load)

% *** MISC UTILITY PACKAGES ***
%
%\usepackage{ifpdf}
% Heiko Oberdiek's ifpdf.sty is very useful if you need conditional
% compilation based on whether the output is pdf or dvi.
% usage:
% \ifpdf
%   % pdf code
% \else
%   % dvi code
% \fi
% The latest version of ifpdf.sty can be obtained from:
% http://www.ctan.org/pkg/ifpdf
% Also, note that IEEEtran.cls V1.7 and later provides a builtin
% \ifCLASSINFOpdf conditional that works the same way.
% When switching from latex to pdflatex and vice-versa, the compiler may
% have to be run twice to clear warning/error messages.

% *** CITATION PACKAGES ***
%
%\usepackage{cite}
% cite.sty was written by Donald Arseneau
% V1.6 and later of IEEEtran pre-defines the format of the cite.sty package
% \cite{} output to follow that of the IEEE. Loading the cite package will
% result in citation numbers being automatically sorted and properly
% "compressed/ranged". e.g., [1], [9], [2], [7], [5], [6] without using
% cite.sty will become [1], [2], [5]--[7], [9] using cite.sty. cite.sty's
% \cite will automatically add leading space, if needed. Use cite.sty's
% noadjust option (cite.sty V3.8 and later) if you want to turn this off
% such as if a citation ever needs to be enclosed in parenthesis.
% cite.sty is already installed on most LaTeX systems. Be sure and use
% version 5.0 (2009-03-20) and later if using hyperref.sty.
% The latest version can be obtained at:
% http://www.ctan.org/pkg/cite
% The documentation is contained in the cite.sty file itself.

% *** GRAPHICS RELATED PACKAGES ***
%
\ifCLASSINFOpdf
  \usepackage[pdftex]{graphicx}
  \usepackage[nocompress]{cite}
  \usepackage{amsmath}
  \usepackage{adjustbox}
  \usepackage{amssymb}
  \usepackage{booktabs}
  \usepackage{algorithm}
  \usepackage{silence}
  \WarningFilter{caption}{Unknown document class}
  \usepackage[compatibility=false]{caption}
  \usepackage{algorithmic}
  \usepackage{multirow}
  \usepackage{url}
  \usepackage[dvipsnames,table,xcdraw]{xcolor}
% \usepackage{colortbl}
% \usepackage{color}
  \usepackage{tikz}
  \usepackage{threeparttable}
  \usepackage{times}
  \usepackage{subcaption}
  \usepackage{enumitem}
  % \usepackage{subfigure}
  % declare the path(s) where your graphic files are
  % \graphicspath{{../pdf/}{../jpeg/}}
  % and their extensions so you won't have to specify these with
  % every instance of \includegraphics
  % \DeclareGraphicsExtensions{.pdf,.jpeg,.png}
\else
  % or other class option (dvipsone, dvipdf, if not using dvips). graphicx
  % will default to the driver specified in the system graphics.cfg if no
  % driver is specified.
  % \usepackage[dvips]{graphicx}
  % declare the path(s) where your graphic files are
  % \graphicspath{{../eps/}}
  % and their extensions so you won't have to specify these with
  % every instance of \includegraphics
  % \DeclareGraphicsExtensions{.eps}
\fi
% graphicx was written by David Carlisle and Sebastian Rahtz. It is
% required if you want graphics, photos, etc. graphicx.sty is already
% installed on most LaTeX systems. The latest version and documentation
% can be obtained at: 
% http://www.ctan.org/pkg/graphicx
% Another good source of documentation is "Using Imported Graphics in
% LaTeX2e" by Keith Reckdahl which can be found at:
% http://www.ctan.org/pkg/epslatex
%
% latex, and pdflatex in dvi mode, support graphics in encapsulated
% postscript (.eps) format. pdflatex in pdf mode supports graphics
% in .pdf, .jpeg, .png and .mps (metapost) formats. Users should ensure
% that all non-photo figures use a vector format (.eps, .pdf, .mps) and
% not a bitmapped formats (.jpeg, .png). The IEEE frowns on bitmapped formats
% which can result in "jaggedy"/blurry rendering of lines and letters as
% well as large increases in file sizes.
%
% You can find documentation about the pdfTeX application at:
% http://www.tug.org/applications/pdftex

% *** MATH PACKAGES ***
%
%\usepackage{amsmath}
% A popular package from the American Mathematical Society that provides
% many useful and powerful commands for dealing with mathematics.
%
% Note that the amsmath package sets \interdisplaylinepenalty to 10000
% thus preventing page breaks from occurring within multiline equations. Use:
%\interdisplaylinepenalty=2500
% after loading amsmath to restore such page breaks as IEEEtran.cls normally
% does. amsmath.sty is already installed on most LaTeX systems. The latest
% version and documentation can be obtained at:
% http://www.ctan.org/pkg/amsmath

% *** SPECIALIZED LIST PACKAGES ***
%
%\usepackage{algorithmic}
% algorithmic.sty was written by Peter Williams and Rogerio Brito.
% This package provides an algorithmic environment fo describing algorithms.
% You can use the algorithmic environment in-text or within a figure
% environment to provide for a floating algorithm. Do NOT use the algorithm
% floating environment provided by algorithm.sty (by the same authors) or
% algorithm2e.sty (by Christophe Fiorio) as the IEEE does not use dedicated
% algorithm float types and packages that provide these will not provide
% correct IEEE style captions. The latest version and documentation of
% algorithmic.sty can be obtained at:
% http://www.ctan.org/pkg/algorithms
% Also of interest may be the (relatively newer and more customizable)
% algorithmicx.sty package by Szasz Janos:
% http://www.ctan.org/pkg/algorithmicx

% *** ALIGNMENT PACKAGES ***
%
%\usepackage{array}
% Frank Mittelbach's and David Carlisle's array.sty patches and improves
% the standard LaTeX2e array and tabular environments to provide better
% appearance and additional user controls. As the default LaTeX2e table
% generation code is lacking to the point of almost being broken with
% respect to the quality of the end results, all users are strongly
% advised to use an enhanced (at the very least that provided by array.sty)
% set of table tools. array.sty is already installed on most systems. The
% latest version and documentation can be obtained at:
% http://www.ctan.org/pkg/array

% IEEEtran contains the IEEEeqnarray family of commands that can be used to
% generate multiline equations as well as matrices, tables, etc., of high
% quality.

% *** SUBFIGURE PACKAGES ***
%\ifCLASSOPTIONcompsoc
%  \usepackage[caption=false,font=normalsize,labelfont=sf,textfont=sf]{subfig}
%\else
%  \usepackage[caption=false,font=footnotesize]{subfig}
%\fi
% subfig.sty, written by Steven Douglas Cochran, is the modern replacement
% for subfigure.sty, the latter of which is no longer maintained and is
% incompatible with some LaTeX packages including fixltx2e. However,
% subfig.sty requires and automatically loads Axel Sommerfeldt's caption.sty
% which will override IEEEtran.cls' handling of captions and this will result
% in non-IEEE style figure/table captions. To prevent this problem, be sure
% and invoke subfig.sty's "caption=false" package option (available since
% subfig.sty version 1.3, 2005/06/28) as this is will preserve IEEEtran.cls
% handling of captions.
% Note that the Computer Society format requires a larger sans serif font
% than the serif footnote size font used in traditional IEEE formatting
% and thus the need to invoke different subfig.sty package options depending
% on whether compsoc mode has been enabled.
%
% The latest version and documentation of subfig.sty can be obtained at:
% http://www.ctan.org/pkg/subfig

% *** FLOAT PACKAGES ***
%
%\usepackage{fixltx2e}
% fixltx2e, the successor to the earlier fix2col.sty, was written by
% Frank Mittelbach and David Carlisle. This package corrects a few problems
% in the LaTeX2e kernel, the most notable of which is that in current
% LaTeX2e releases, the ordering of single and double column floats is not
% guaranteed to be preserved. Thus, an unpatched LaTeX2e can allow a
% single column figure to be placed prior to an earlier double column
% figure.
% Be aware that LaTeX2e kernels dated 2015 and later have fixltx2e.sty's
% corrections already built into the system in which case a warning will
% be issued if an attempt is made to load fixltx2e.sty as it is no longer
% needed.
% The latest version and documentation can be found at:
% http://www.ctan.org/pkg/fixltx2e

%\usepackage{stfloats}
% stfloats.sty was written by Sigitas Tolusis. This package gives LaTeX2e
% the ability to do double column floats at the bottom of the page as well
% as the top. (e.g., "\begin{figure*}[!b]" is not normally possible in
% LaTeX2e). It also provides a command:
%\fnbelowfloat
% to enable the placement of footnotes below bottom floats (the standard
% LaTeX2e kernel puts them above bottom floats). This is an invasive package
% which rewrites many portions of the LaTeX2e float routines. It may not work
% with other packages that modify the LaTeX2e float routines. The latest
% version and documentation can be obtained at:
% http://www.ctan.org/pkg/stfloats
% Do not use the stfloats baselinefloat ability as the IEEE does not allow
% \baselineskip to stretch. Authors submitting work to the IEEE should note
% that the IEEE rarely uses double column equations and that authors should try
% to avoid such use. Do not be tempted to use the cuted.sty or midfloat.sty
% packages (also by Sigitas Tolusis) as the IEEE does not format its papers in
% such ways.
% Do not attempt to use stfloats with fixltx2e as they are incompatible.
% Instead, use Morten Hogholm'a dblfloatfix which combines the features
% of both fixltx2e and stfloats:
%
% \usepackage{dblfloatfix}
% The latest version can be found at:
% http://www.ctan.org/pkg/dblfloatfix

%\ifCLASSOPTIONcaptionsoff
%  \usepackage[nomarkers]{endfloat}
% \let\MYoriglatexcaption\caption
% \renewcommand{\caption}[2][\relax]{\MYoriglatexcaption[#2]{#2}}
%\fi
% endfloat.sty was written by James Darrell McCauley, Jeff Goldberg and 
% Axel Sommerfeldt. This package may be useful when used in conjunction with 
% IEEEtran.cls'  captionsoff option. Some IEEE journals/societies require that
% submissions have lists of figures/tables at the end of the paper and that
% figures/tables without any captions are placed on a page by themselves at
% the end of the document. If needed, the draftcls IEEEtran class option or
% \CLASSINPUTbaselinestretch interface can be used to increase the line
% spacing as well. Be sure and use the nomarkers option of endfloat to
% prevent endfloat from "marking" where the figures would have been placed
% in the text. The two hack lines of code above are a slight modification of
% that suggested by in the endfloat docs (section 8.4.1) to ensure that
% the full captions always appear in the list of figures/tables - even if
% the user used the short optional argument of \caption[]{}.
% IEEE papers do not typically make use of \caption[]'s optional argument,
% so this should not be an issue. A similar trick can be used to disable
% captions of packages such as subfig.sty that lack options to turn off
% the subcaptions:
% For subfig.sty:
% \let\MYorigsubfloat\subfloat
% \renewcommand{\subfloat}[2][\relax]{\MYorigsubfloat[]{#2}}
% However, the above trick will not work if both optional arguments of
% the \subfloat command are used. Furthermore, there needs to be a
% description of each subfigure *somewhere* and endfloat does not add
% subfigure captions to its list of figures. Thus, the best approach is to
% avoid the use of subfigure captions (many IEEE journals avoid them anyway)
% and instead reference/explain all the subfigures within the main caption.
% The latest version of endfloat.sty and its documentation can obtained at:
% http://www.ctan.org/pkg/endfloat
%
% The IEEEtran \ifCLASSOPTIONcaptionsoff conditional can also be used
% later in the document, say, to conditionally put the References on a 
% page by themselves.

% *** PDF, URL AND HYPERLINK PACKAGES ***
%
%\usepackage{url}
% url.sty was written by Donald Arseneau. It provides better support for
% handling and breaking URLs. url.sty is already installed on most LaTeX
% systems. The latest version and documentation can be obtained at:
% http://www.ctan.org/pkg/url
% Basically, \url{my_url_here}.

% *** Do not adjust lengths that control margins, column widths, etc. ***
% *** Do not use packages that alter fonts (such as pslatex).         ***
% There should be no need to do such things with IEEEtran.cls V1.6 and later.
% (Unless specifically asked to do so by the journal or conference you plan
% to submit to, of course. )

% correct bad hyphenation here
\hyphenation{op-tical net-works semi-conduc-tor}

\markboth{IEEE Transactions on Biometrics, Behavior, and Identity Science}%
{Pujari and Rattani: AudioAuth}

\begin{document}
%
% paper title
% Titles are generally capitalized except for words such as a, an, and, as,
% at, but, by, for, in, nor, of, on, or, the, to and up, which are usually
% not capitalized unless they are the first or last word of the title.
% Linebreaks \\ can be used within to get better formatting as desired.
% Do not put math or special symbols in the title.
\title{AudioAuth: A Dual-Watermarking Framework for Robust Audio Integrity and Source Attribution}

\author{Aditya Pujari,
        Ajita Rattani% <-this % stops a space
\thanks{Aditya Pujari and Ajita Rattani are with the Computer Science and Engineering Department, University of North Texas, Denton, TX 76203, USA. (E-mail: Adityapujari@my.unt.edu, Ajita.Rattani@unt.edu)}% <-this % stops a space
}

% note the % following the last \IEEEmembership and also \thanks - 
% these prevent an unwanted space from occurring between the last author name
% and the end of the author line. i.e., if you had this:
% 
% \author{....lastname \thanks{...} \thanks{...} }
%                     ^------------^------------^----Do not want these spaces!
%
% a space would be appended to the last name and could cause every name on that
% line to be shifted left slightly. This is one of those "LaTeX things". For
% instance, "\textbf{A} \textbf{B}" will typeset as "A B" not "AB". To get
% "AB" then you have to do: "\textbf{A}\textbf{B}"
% \thanks is no different in this regard, so shield the last } of each \thanks
% that ends a line with a % and do not let a space in before the next \thanks.
% Spaces after \IEEEmembership other than the last one are OK (and needed) as
% you are supposed to have spaces between the names. For what it is worth,
% this is a minor point as most people would not even notice if the said evil
% space somehow managed to creep in.

% The paper headers
% The only time the second header will appear is for the odd numbered pages
% after the title page when using the twoside option.
% 
% *** Note that you probably will NOT want to include the author's ***
% *** name in the headers of peer review papers.                   ***
% You can use \ifCLASSOPTIONpeerreview for conditional compilation here if
% you desire.

% If you want to put a publisher's ID mark on the page you can do it like
% this:
%\IEEEpubid{0000--0000/00\$00.00~\copyright~2015 IEEE}
% Remember, if you use this you must call \IEEEpubidadjcol in the second
% column for its text to clear the IEEEpubid mark.

% use for special paper notices
%\IEEEspecialpapernotice{(Invited Paper)}

% make the title area
\maketitle

% \begin{abstract}
% The rapid advancement of neural speech synthesis has enabled synthetic speech spoofing attacks, including AI-driven speaker impersonation, threatening voice biometric systems. These attacks compromise both the integrity of speech presented to speaker verification and reliable attribution of synthetic speech sources. While audio watermarking offers a proactive defense for provenance verification and tamper detection, existing methods remain vulnerable to spectral-domain attacks, provide coarse localization, and degrade under enhancement, vocoding, or compression. We propose AudioAuth, a dual-watermarking framework for joint content integrity verification and source attribution in voice biometric security. AudioAuth employs a frequency-partitioned Feature-wise Linear Modulation (FiLM) generator with dual-channel embedding: a fixed 16-bit model identifier embedded in even frequency bands enables robust source attribution, while a dynamic 16-bit payload embedded in odd bands supports fine-grained authentication and sub-second tamper localization. A dynamic effect scheduler enables co-adaptive generator--detector training for robustness to adversarial distortions. AudioAuth outperforms six neural watermarking baselines on ASVspoof 2019 and RAVDESS, achieving precise temporal localization (mean intersection-over-union (MIoU) of 0.984 under high-pass filtering), near-zero bit error rate under MP3 compression (below 0.1\%), robustness to temporal desynchronization (e.g., time-scale modification, cropping), and high perceptual quality (short-time objective intelligibility 0.96, speech quality score 4.76). Evaluation on physically replayed deepfakes (ReplayDF) demonstrates consistent detection across 109 loudspeaker--microphone configurations (average MIoU 0.958), accurate multi-source text-to-speech (TTS) attribution (98.8\% across eight synthesis systems), and negligible impact on speaker verification (equal error rate increase of 0.07\%). These results establish AudioAuth as a practical solution for protecting voice biometric systems against synthetic speech attacks.
% \end{abstract}

% \begin{IEEEkeywords}
% Audio watermarking, speaker verification, spoofing attacks, synthetic speech attribution, voice biometric security
% \end{IEEEkeywords}

\begin{abstract}
Voice biometric systems face escalating threats from neural speech synthesis technologies capable of producing speaker-indistinguishable audio. Existing audio watermarking defenses remain vulnerable to spectral-domain attacks, offer only coarse tamper localization, and often degrade under codec compression.
We propose AudioAuth, a dual-watermarking framework for joint content integrity verification and source attribution in voice biometric security, extending our prior single-channel work WaveVerify. AudioAuth employs a frequency-partitioned dual-channel encoding strategy: a fixed model identifier embedded in even frequency bands enables reliable source attribution, while a dynamic payload embedded in odd bands supports fine-grained temporal tamper localization. A dynamic effect scheduler further enables co-adaptive training, improving robustness against adversarial distortions and signal transformations.
Evaluated against six state-of-the-art baselines spanning signal-level (AudioSeal, WavMark, WaveVerify) and representation-level (WMCodec, Timbre, DiscreteWM) approaches, AudioAuth achieves precise temporal localization (MIoU = $0.983$), accurate multi-source TTS attribution (98.8\% across eight systems), and negligible impact on speaker verification performance (EER increase of only $0.07\%$).
\end{abstract}

\begin{IEEEkeywords}
Audio watermarking, speaker verification, synthetic speech attacks, synthetic speech attribution, voice biometric security
\end{IEEEkeywords}
\IEEEpeerreviewmaketitle

\section{Introduction}

% \IEEEPARstart{T}{he} security of voice biometric authentication pipelines increasingly depends on proactive defenses that embed verifiable provenance directly during speech generation. This paper presents AudioAuth, a dual-channel watermarking framework that embeds model-attribution and content-integrity signals within text-to-speech (TTS) pipelines, evaluated under biometric security conditions including physically replayed deepfakes across 109 acoustic configurations, multi-source attribution over eight TTS systems, open-set verification, and speaker verification impact. The motivation for this framework stems from the rapid advancement of neural speech generation, which has enabled synthesis perceptually close to bonafide human speech~\cite{valle,speartts}. While these technologies support beneficial applications such as personalized TTS~\cite{yan1} and voice preservation~\cite{voicepreservation}, they also introduce growing risks to \emph{voice biometric systems}, particularly speaker verification pipelines used for identity authentication~\cite{tomi2021tandem,das2020assessing}.

\IEEEPARstart{T}{he} rapid advancement of neural speech generation has enabled voice synthesis that is perceptually close to bona fide human speech~\cite{valle,speartts}. While these technologies support beneficial applications such as personalized text-to-speech (TTS) and voice preservation, they also introduce growing risks to \emph{voice biometric systems}, particularly speaker verification pipelines used for identity authentication~\cite{tomi2021tandem,das2020assessing}.

Automatic Speaker Verification (ASV) systems--core components of modern voice biometric authentication for banking, access control, and identity verification--are increasingly exposed to \emph{synthetic speech--based spoofing attacks}, including replay, text-to-speech, and voice conversion~\cite{asvspoof2017,asvspoof2019,asvspoof2021}. Industry analyses report a rapid rise in AI-enabled impersonation attacks, with deepfake (synthetic speech) fraud attempts increasing by over $1{,}300\%$ in 2024~\cite{pindrop2025}. The rapid proliferation of synthetic and replay-based speech attacks directly undermines the security and reliability of voice biometric systems deployed in security-critical settings. Complementing these real-world observations, standardized evaluations through the ASVspoof challenge series have systematically quantified ASV vulnerabilities across diverse acoustic and channel conditions, demonstrating that even state-of-the-art systems remain susceptible. These findings underscore the need for robust and deployable provenance verification mechanisms that can reliably associate speech content with its generative source and detect malicious modifications—capabilities essential for protecting voice biometric security pipelines.

A common forensic mitigation strategy is \emph{passive detection}, in which classifiers are trained to discriminate between bona fide and synthetic speech~\cite{xiao2024xlsr,zhang2024audio}. While appealing due to its low deployment overhead, this approach faces inherent generalization challenges as generative models continue to narrow perceptual and statistical gaps with real speech. Detectors trained on earlier synthesis systems (e.g., Tacotron~2~\cite{tacotron2}) often fail to generalize to newer architectures, including diffusion-based and autoregressive models~\cite{DiffWave,valle}. 

\textbf{Audio Watermarking and Model Attribution.}
In contrast to passive detection, \emph{audio watermarking} embeds imperceptible, cryptographically bound signals directly into speech at generation time, enabling persistent provenance verification and source attribution even under common signal transformations~\cite{audioseal,wavmark}. Modern \emph{neural audio watermarking} advances beyond classical spread-spectrum, patchwork, and echo-hiding techniques~\cite{cox1997secure,2003patchwork} by jointly optimizing embedder--detector pairs under simulated distortions during training~\cite{zhu2018hidden,EMNLP2024}.
Existing neural watermarking methods broadly fall into two paradigms: \emph{signal-level approaches}, which embed perturbations directly into the waveform (e.g., WavMark, AudioSeal, WaveVerify), and \emph{representation-level approaches}, which encode watermarks within learned latent spaces (e.g., DiscreteWM, Timbre Watermarking, WMCodec). Despite recent progress, existing watermarking approaches remain ill-suited for biometric deployments: signal-level methods are vulnerable to time--frequency distortions and temporal desynchronization, while representation-level methods trade robustness for generalization due to tight coupling with specific codecs or synthesis architectures~\cite{cox1997secure,bassia2001robust,liu2024audiomarkbench,cox2002digital,li2024syncguard,SoK2024}.

Beyond watermark detection for content integrity verification, \emph{model attribution} seeks to associate speech with its generative source. Recent attribution methods such as FakeMark~\cite{ge2025fakemark} and AudioMarkNet~\cite{zong2025audiomarknet} encode model-dependent signatures using synthesis artifacts or by exploiting watermark survivability through voice cloning pipelines. While effective under constrained conditions, these approaches rely on fragile model-specific cues that are often suppressed by post-processing operations such as speech enhancement, neural vocoding, and codec compression, limiting attribution reliability. Moreover, attribution-focused methods do not support \emph{speech content integrity verification}, as they cannot detect or localize modifications applied after watermark embedding.

%In a typical voice biometric pipeline—call-center authentication, mobile banking, or VoIP identity verification—incoming speech must pass through a provenance gate before reaching the speaker verification module. The gate must answer two questions in sequence: \emph{which model generated this utterance}, and \emph{has the content been altered since generation}—for instance, by splicing, partial re-synthesis, or selective enhancement aimed at defeating anti-spoofing checks. Samples that fail either check are flagged for manual review rather than forwarded to the verifier. Meeting this requirement demands a single embedded watermark that carries model identity and supports fine-grained tamper localization while remaining recoverable after codec compression and the channel distortions typical of telephony and streaming deployments.

\begin{figure*}[t]
  \centering
  \begin{minipage}[c]{0.68\textwidth}
    \centering
    \includegraphics[height=6cm, width=\textwidth, keepaspectratio]{IEEEtran/figures/new_architecture.png}
  \end{minipage}
  \hfill
  \begin{minipage}[c]{0.28\textwidth}
    \centering
    \includegraphics[height=6cm, width=\textwidth, keepaspectratio]{IEEEtran/figures/detector_architecture.png}
  \end{minipage}
  \caption{
      End-to-end AudioAuth pipeline.
      \textbf{(Left)} System overview with panels labeled (a)--(d):
      \textbf{(a)}~FiLM-based generator with frequency-partitioned dual watermark embedding across hierarchical encoder layers;
      \textbf{(b)}~Dynamic augmentation scheduler applying temporal and audio effect transformations;
      \textbf{(c)}~Dual-network extraction architecture with detector for bit recovery and locator for temporal boundary identification;
      \textbf{(d)}~Multi-scale discriminators providing adversarial feedback to enforce perceptual quality of the watermarked audio.
      \textbf{(Right)} Detector architecture: Encoder processes watermarked speech through Conv1D, SpecBlock, and stacked Residual Blocks, followed by reverse convolution (ConvTranspose1d) to extract the 16-bit model watermark and 16-bit data watermark.
  }
  \label{fig:end-to-end-architecture}
\end{figure*}

\textbf{Dual Watermarking.}
Prior work typically treats source attribution and speech content integrity verification as separate objectives~\cite{audioseal,wavmark,ge2025fakemark,zong2025audiomarknet,ji2025discretewm,liu2024timbrewm,wmcodec2025}, limiting applicability in biometric security settings where both capabilities must be supported simultaneously. \emph{Dual watermarking} addresses this gap by embedding attribution and integrity signals within the same audio under a unified framework. However, jointly embedding multiple watermark signals introduces new adversarial \textbf{challenges} not encountered in single-objective designs. First, \emph{coordinated dual-path attacks} (Challenge~1) allow adversaries to chain temporal and spectral distortions that progressively degrade each embedded signal, undermining the intended redundancy of dual watermarking. Second, \emph{frequency-selective targeting} (Challenge~2)\footnote{The terms ``spectral attacks'' and ``frequency-selective attacks'' are used interchangeably throughout this paper.} exploits band-limited suppression or notch filtering to selectively disrupt one watermark component while leaving the other intact. Third, \emph{cross-signal desynchronization} (Challenge~3) arises when temporal manipulations such as time-stretching or jitter misalign embedded signals, reducing joint recoverability. Fourth, \emph{codec-driven signal suppression} (Challenge~4) occurs when neural codecs quantize or discard watermark energy in perceptually less salient regions, erasing one signal while preserving the other. Finally, \emph{adaptive attack composition} (Challenge~5) involves sequential, adversary-aware distortion pipelines that exploit detector weaknesses across channels, enabling selective suppression of attribution or integrity signals through strategic attack ordering.

\textbf{Our Contributions.}
We propose \textbf{AudioAuth}\footnote{Code and models are publicly available at: \url{https://github.com/vcbsl/AudioAuth}}, a dual-channel audio watermarking framework that jointly embeds \emph{model attribution} and \emph{speech content integrity verification} signals within a unified representation. Unlike prior work that treats these objectives independently, AudioAuth is explicitly designed to withstand the adversarial challenges inherent to dual watermarking.

AudioAuth employs a frequency-partitioned Feature-wise Linear Modulation (FiLM) generator with complementary subband weighting, in which model-identification bits and integrity bits are interleaved across even and odd frequency bands (70/30 split). This design ensures that distortions targeting specific spectral regions preserve recoverable watermark information in adjacent bands, directly mitigating coordinated spectral and temporal attacks (Challenges~1--4). To address adaptive, multi-stage attacks (Challenge~5), AudioAuth incorporates adversarial training with a dynamic attack scheduler that prioritizes compound distortions based on real-time detection difficulty. At inference, a two-stage architecture decouples localization and message recovery using a lightweight \emph{locator} for sample-level boundary detection and a robust \emph{detector} for confidence-guided bit extraction, enabling precise temporal localization while remaining resilient to desynchronization and partial removal.

\textit{Deployment Scenario for Voice Biometrics.} AudioAuth is designed as a \textbf{proactive provenance layer} for voice biometric systems, operating under the realistic assumption that synthetic speech is watermarked at generation time, while bona fide human speech is not. Specifically, AudioAuth adopts a proactive watermarking paradigm, embedding provenance information directly within authorized text-to-speech (TTS) or voice conversion pipelines at the point of synthesis. In identity authentication pipelines, AudioAuth acts as a front-end filter to Automatic Speaker Verification (ASV), where the presence of a valid watermark indicates synthetic or replayed speech and triggers rejection or further scrutiny, while unwatermarked inputs are processed normally by ASV. This design enables reliable detection of synthetic, cloned, or manipulated speech without requiring watermarking of human audio, and supports post-hoc forensic analysis through source attribution and tamper localization.


% \textit{Scope of ``Authentication.''}
% The term \emph{authentication} in the title of this paper encompasses three distinct functions that AudioAuth supports jointly. \textbf{Presence detection} determines whether a watermark exists in a given audio segment, thereby establishing provenance: the presence of a valid watermark confirms synthetic origin and triggers further scrutiny by downstream systems. \textbf{Source attribution} decodes the 16-bit model-identifier payload to identify the specific TTS or voice conversion system that generated the speech, enabling forensic tracing across multiple generative sources (Table~IV). \textbf{Integrity verification} leverages the 16-bit data-authentication payload together with the temporal locator to detect and localize content modifications at the sample level, quantifying the extent and location of tampering (Tables~I--III). Throughout this paper, ``authentication'' refers to the union of all three functions; where a specific function is intended, the corresponding term is used explicitly.
\textit{Scope of AudioAuth.}
The AudioAuth framework supports \emph{three} watermark-based functions for audio provenance verification in voice biometric settings.
\textit{Watermark detection} determines whether a watermark exists in an audio segment, indicating synthetic origin and triggering downstream scrutiny in identity authentication pipelines.
\textit{Source attribution} decodes the 16-bit model-identifier payload to identify the TTS or voice conversion system responsible for generating the speech, enabling forensic tracing (Table IV).
\textit{Content-integrity verification} or integrity verification uses the 16-bit data-integrity payload with a temporal locator to detect and localize modifications at the sample level, quantifying the extent and location of tampering (Tables I–III).
Throughout this paper, these functions are referred to by their specific names. 
%.; to avoid any confusion, the term \emph{authentication} is used solely in the context of biometric user authentication.
%The AudioAuth framework supports three distinct watermark-based functions that jointly enable audio provenance verification in voice biometric settings. \textbf{Watermark presence detection} determines whether a watermark exists in a given audio segment, thereby establishing provenance: the presence of a valid watermark confirms synthetic origin and triggers further scrutiny by downstream identity authentication systems. \textbf{Source attribution} decodes the 16-bit model-identifier payload to identify the specific TTS or voice conversion system that generated the speech, enabling forensic tracing across multiple generative sources (Table~IV). \textbf{Content-integrity verification} leverages the 16-bit data-integrity payload together with the temporal locator to detect and localize content modifications at the sample level, quantifying the extent and location of tampering (Tables~I--III). Throughout this paper, these three watermark functions are referred to by their specific names; the term \emph{authentication}, when used without qualification, refers to identity authentication in the context of voice biometric systems.

The main \textbf{contributions} of this work are:

\begin{enumerate}
     \item \textbf{Dual-channel watermarking for voice biometric provenance.} A FiLM-conditioned generator that jointly embeds model-attribution and speech-integrity signals through complementary frequency partitioning, enabling simultaneous source identification and tamper detection for TTS-generated biometric speech under spectral filtering, temporal manipulation, and codec transforms.

     \item \textbf{Decoupled localization and detection architecture.} A two-stage extraction framework separating sample-level watermark boundary detection from bit-level message recovery, improving resilience to temporal desynchronization while enabling fine-grained analysis of manipulated speech segments.

     \item \textbf{Adversarial training against coordinated attacks.} A dynamic training strategy that co-adapts the embedder and detector to compound, multi-stage attack chains—including those targeting voice biometric pipelines—improving generalization beyond fixed augmentation regimes.

     \item \textbf{Comprehensive evaluation under biometric security conditions.} Extensive experiments spanning physically replayed deepfakes across 109 loudspeaker--microphone configurations, multi-source attribution over eight TTS systems, open-set watermark verification, and speaker verification impact analysis demonstrate state-of-the-art robustness, high imperceptibility, reliable provenance tracing, and negligible degradation of downstream ASV performance.
\end{enumerate}

\textbf{Contributions Beyond WaveVerify~\cite{waveverify2025}.}
This work substantially extends our prior conference paper WaveVerify~\cite{waveverify2025} published at IEEE IJCB~2025, with new methodological developments and a focused treatment of voice biometric security. While WaveVerify introduced FiLM-based multiband watermark embedding, AudioAuth advances this framework in several key dimensions:

\begin{enumerate}[leftmargin=*, nosep]
\item \textbf{Dual-channel frequency-partitioned embedding.}
Unlike WaveVerify’s uniform weighting across frequency bands, AudioAuth jointly embeds model-identification and speech content integrity payloads using complementary 70/30 subband weighting, providing inherent path diversity against frequency-selective attacks relevant to speaker verification systems.

% \item \textbf{Attack-aware training with dynamic composition.}
% AudioAuth introduces a dynamic adversarial training strategy that models sequential and compound distortion chains (e.g., filtering followed by compression), extending beyond WaveVerify’s fixed augmentation regime and This extends beyond WaveVerify’s fixed augmentation regime and significantly improves robustness to adaptive, multi-stage attacks aligned with Challenges~1--5 associated with Dual-watermarking.

\item \textbf{Attack-aware training with dynamic composition.}
AudioAuth extends WaveVerify's augmentation catalog with neural codec compression, compound distortion chains (e.g., filtering followed by compression), and dual-channel co-adaptation, significantly improving robustness to adaptive, multi-stage attacks (Challenge~5).

\item \textbf{Decoupled detection and localization.}
A two-stage extraction architecture separates sample-level watermark localization from bit-level message recovery, enabling precise tamper localization and improved robustness under partial removal and temporal desynchronization. Unlike WaveVerify, which relies on a single detection network that jointly performs localization and bit extraction, AudioAuth introduces a dedicated lightweight locator module (0.13\,M parameters) for per-sample boundary detection, entirely absent from WaveVerify's architecture. This decoupling enables independent optimization of each sub-task: the locator achieves sample-level temporal precision while the detector focuses on robust message recovery under distortion.


\item \textbf{Expanded evaluation of voice biometric security.}
New experiments evaluate robustness under ASVspoof~2019 physical access replay conditions, multi-source TTS attribution, open-set watermark verification, and the impact on speaker verification performance, alongside ablations validating the proposed frequency-partitioning design.
\end{enumerate}

\section{Related Work}\label{sec:relatedwork}
Protecting voice biometric pipelines against synthetic speech injection requires provenance mechanisms robust to the diverse signal transformations of modern speech processing. Audio watermarking has evolved through three paradigms, each with distinct robustness–capacity trade-offs: classical signal-processing methods are resilient to analog distortions but have very low capacity (${ \sim }1$ bps) and poor robustness to neural codecs; neural signal-level approaches improve capacity (16–32 bps) but remain vulnerable to temporal desynchronization and partial removal; and neural representation-level methods achieve codec-coupled robustness but generalize poorly across architectures and pipelines.
These trade-offs expose \emph{four key adversarial challenges} for dual-watermark provenance in voice biometric systems—coordinated dual-path attacks, frequency-selective targeting, cross-signal desynchronization, and codec-adaptive resilience (Section~I)—which we evaluate in the following sections.

\subsection{Traditional Audio Watermarking}
Classical spread-spectrum and echo-hiding schemes~\cite{cox1997secure,boney1996digital}, along with patchwork and phase-coding extensions~\cite{arnold2003phase,2003patchwork}, embed watermarks via signal-domain energy distribution.
Modern neural audio watermarking adopts an encoder–decoder paradigm from image steganography, notably HiDDeN~\cite{zhu2018hidden}, adapted to the temporal audio domain.
Despite psychoacoustic masking, these approaches remain brittle under compression, resampling, and temporal desynchronization~\cite{cox2002digital,bassia2001robust}, motivating jointly learned embedding–detection architectures. In voice biometric pipelines subject to telephony compression and re-encoding, such brittleness makes classical watermarks unsuitable for provenance.

\subsection{Neural Audio Watermarking}
\subsubsection{End-to-End Signal-Level Perturbation Methods}
WavMark~\cite{wavmark} achieves high payload capacity (up to 32 bits/s) and accurate localization via explicit synchronization patterns, but its exhaustive offset search at inference incurs computational overhead incompatible with real-time biometric verification. AudioSeal~\cite{audioseal} enables efficient per-frame scoring, yet its frame-dependent design lacks temporal invariance: common telephony distortions (e.g., time-scale modification and packet-loss concealment) disrupt frame alignment, causing severe degradation under realistic biometric traffic (Table~II).

WaveVerify~\cite{waveverify2025} improves robustness through FiLM-conditioned multiband embedding, offering resilience to frequency-selective attacks (Challenge~2). However, its single-channel, equal-weight band allocation conflates attribution and integrity signals, precluding independent verification—an important limitation for fine-grained voice biometric provenance. IDEAW~\cite{EMNLP2024} introduces dual embedding via invertible networks, but encodes redundant payloads rather than semantically distinct attribution and integrity signals, and lacks evaluation under compound biometric distortions.
Despite these advances, signal-level watermarking remains vulnerable to neural codecs and denoisers. Recent studies show that shallow networks can remove post-hoc watermarks while preserving speech quality~\cite{oreilly2025shallow}, and that none of the evaluated schemes survives neural codec processing due to spectral competition~\cite{ozer2025comprehensive}. Moreover, existing methods do not address coordinated dual-path attacks or cross-signal desynchronization (Challenges~1,~3), nor do they jointly embed attribution and integrity signals. For speaker verification systems, this gap allows attackers to strip provenance while retaining spoofing capability—motivating watermarking strategies with codec-aware robustness.
%WavMark~\cite{wavmark} employs explicit synchronization patterns to achieve high payload capacity (up to 32 bits/s) and accurate localization; however, its exhaustive cross-correlation offset search at inference incurs computational overhead that is prohibitive for real-time biometric verification pipelines. AudioSeal~\cite{audioseal} enables efficient per-frame likelihood scoring, but its frame-aligned architecture lacks temporal invariance: common telephony distortions such as time-scale modification and packet-loss concealment disrupt frame boundaries, leading to sharp performance degradation under conditions routinely encountered in voice biometric traffic (Table~II).

%WaveVerify~\cite{waveverify2025} improves robustness by distributing watermark energy across FiLM-conditioned frequency bands, offering stronger resilience to frequency-selective attacks (Challenge~2) than uniform embedding. However, its equal-weight band allocation and single shared detection head restrict it to a single 16-bit channel, conflating model attribution and content integrity signals and preventing independent verification of each—an important limitation for voice biometric deployments requiring fine-grained provenance guarantees. IDEAW~\cite{EMNLP2024} introduces invertible dual embedding for neural audio watermarking, but its dual channels encode redundant replicas of the same payload for error correction rather than semantically distinct attribution and integrity signals, and its invertible architecture has not been evaluated under compound biometric-pipeline distortions.

%Despite these advances, signal-level watermarking methods remain fundamentally vulnerable to neural codecs and denoisers that suppress perceptually inconspicuous perturbations. O’Reilly et al.~\cite{oreilly2025shallow} show that shallow neural networks can effectively remove state-of-the-art post-hoc audio watermarks while preserving speech quality, exposing a structural weakness exploitable by targeted removal. Ozer et al.~\cite{ozer2025comprehensive} further corroborate this fragility: RAW-Bench evaluations reveal that none of the tested watermarking schemes survives neural codec processing, as codecs and watermarks compete for the same perceptually salient spectral regions. Additional vulnerabilities arise under compound distortion pipelines (e.g., time-scale modification followed by denoising and re-encoding), temporal drift, cropping, and physical re-recording~\cite{cox2002digital,bassia2001robust}.

%Critically, existing signal-level methods do not address coordinated dual-path attacks or cross-signal desynchronization (Challenges~1 and~3), nor do they jointly embed model attribution and content integrity signals. For speaker verification systems, this gap allows attackers to strip provenance information from cloned speech while preserving the biometric features required to spoof enrollment templates. These limitations motivate alternative strategies that embed watermarks within learned latent representations, trading deployment flexibility for codec-coupled robustness.

\subsubsection{Representation-Level Watermarking}
DiscreteWM~\cite{ji2025discretewm} embeds messages into vector-quantized tokens, achieving strong robustness and imperceptibility, while Timbre Watermarking~\cite{liu2024timbrewm} targets speaker-centric protection through repeated frequency-domain embedding with attack-aware distortions. These approaches leverage structured latent spaces to separate content from watermark signals. WMCodec~\cite{wmcodec2025} further embeds watermarks directly into a neural codec’s latent bottleneck via cross-attention, tightly coupling verification to a single codec architecture.

However, representation-level methods suffer from codec dependence, limited cross-codec portability, and capacity constraints imposed by latent bottlenecks (e.g., 8--16~bps in DiscreteWM versus AudioAuth's 32~bps). Fixed distortion models further limit generalization to unseen or compound attacks, reducing applicability in real-world multi-codec and archival settings. While latent-space embedding avoids post-hoc perturbation removal during decompression, this robustness comes at the cost of portability and deployment flexibility. These limitations leave codec-driven signal suppression (Challenge~4) unresolved for cross-codec deployment scenarios. In voice biometric contexts, codec dependence is particularly problematic: speech traverses multiple transcoding stages between enrollment and verification, and a watermark that fails cross-codec transfer cannot serve as a reliable provenance signal in multi-platform identity authentication scenarios.

\subsection{Model Attribution Techniques}
A growing body of work extends audio watermarking beyond binary detection toward \emph{attribution}, linking speech to its generative source. Existing approaches bifurcate into artifact-correlation methods and robustness benchmarking efforts. On the correlation side, FakeMark~\cite{ge2025fakemark} ties watermarks to model-specific synthesis artifacts, while AudioMarkNet~\cite{zong2025audiomarknet} embeds watermarks prior to voice cloning and detects their survival post-synthesis, attributing speech to a cloning pipeline rather than verifying post-generation integrity. On the evaluation side, AudioMarkBench~\cite{liu2024audiomarkbench} shows that no current scheme achieves both high attribution accuracy and robustness under compound distortions, and a recent SoK~\cite{SoK2024} identifies a trade-off between per-model and per-user keying.
Orthogonal efforts such as SyncGuard~\cite{li2024syncguard} improve temporal robustness but lack attribution keying, while industry frameworks like C2PA~\cite{c2pa2024} rely on soft-binding metadata without adversarial guarantees. 

Attribution-oriented watermarking also faces detector transferability~\cite{ji2025discretewm} and semantic-loop attacks (ASR$\rightarrow$TTS) that remove signal-level watermarks while preserving content~\cite{liu2024timbrewm,oreilly2025shallow}. Critically, no existing method jointly embeds model attribution and content integrity signals, motivating unified, adversary-aware watermarking frameworks for secure speaker verification.

\subsection{Voice Biometrics and Anti-Spoofing}
Identity authentication in voice biometric systems relies on deep neural speaker embeddings such as x-vectors~\cite{snyder2018xvector} and ECAPA-TDNN~\cite{Desplanques_2020}, and increasingly on self-supervised speech representations (e.g., wav2vec~2.0~\cite{baevski2020wav2vec2}, WavLM~\cite{chen2022wavlm}) that achieve state-of-the-art verification and anti-spoofing performance. While these advances improve accuracy, they also expand the attack surface of modern voice biometric systems.

Automatic Speaker Verification (ASV) systems face diverse spoofing threats. Logical access attacks exploit text-to-speech (TTS) and voice conversion, while physical access attacks replay recorded speech~\cite{asvspoof2017,asvspoof2019,asvspoof2021}. Recent neural codec language models (e.g., VALL-E~\cite{valle}) further exacerbate risk by enabling high-fidelity speaker cloning from minimal enrollment data. Additional threats include adversarial perturbations and presentation attacks targeting liveness detection~\cite{wang2025overtheairadversarialattackdetection,jamdar2025syntheticpopattackingspeakerverification}. Notably, the tandem detection cost function (t-DCF)~\cite{tomi2021tandem} shows that spoofing detection alone is insufficient for end-to-end security.
Although anti-spoofing countermeasures have advanced toward end-to-end neural models~\cite{tak2021end,muller2024generalization}, they remain inherently \emph{passive} and brittle under distribution shift. ASVspoof evaluations consistently show poor generalization to unseen synthesis models~\cite{nautsch2021asvspoof,asvspoof2021}, motivating proactive defenses beyond post-hoc discrimination.

Audio watermarking offers such a proactive channel by embedding verifiable provenance at generation time, enabling ASV systems to assess speech authenticity prior to identity verification. However, the impact of watermark embedding on downstream speaker verification embeddings remains largely unexamined, leaving compatibility with modern ASV pipelines unvalidated.

% \section{Proposed Method}\label{sec:method}

% \begin{figure*}[t]
%   \centering
%   \begin{minipage}[c]{0.68\textwidth}
%     \centering
%     \includegraphics[height=6cm, width=\textwidth, keepaspectratio]{IEEEtran/figures/architecture.png}
%   \end{minipage}
%   \hfill
%   \begin{minipage}[c]{0.28\textwidth}
%     \centering
%     \includegraphics[height=6cm, width=\textwidth, keepaspectratio]{IEEEtran/figures/detector_architecture.png}
%   \end{minipage}
%   \caption{
%       End-to-end AudioAuth pipeline.
%       \textbf{(Left)} System overview with panels labeled (a)--(d):
%       \textbf{(a)}~FiLM-based generator with frequency-partitioned dual watermark embedding across hierarchical encoder layers;
%       \textbf{(b)}~Dynamic augmentation scheduler applying temporal and audio effect transformations;
%       \textbf{(c)}~Dual-network extraction architecture with detector for bit recovery and locator for temporal boundary identification;
%       \textbf{(d)}~Multi-scale discriminators providing adversarial feedback to enforce perceptual quality of the watermarked audio.
%       \textbf{(Right)} Detector architecture: Encoder processes watermarked speech through Conv1D, SpecBlock, and stacked Residual Blocks, followed by reverse convolution (ConvTranspose1d) to extract the 16-bit model watermark and 16-bit data watermark.
%   }
%   \label{fig:end-to-end-architecture}
% \end{figure*}

% \begin{figure}[t]
%   \centering
%   \includegraphics[width=\linewidth, height=12cm, keepaspectratio]{IEEEtran/figures/generator_encoder.png}
%   \caption{Generator encoder architecture. Frequency-partitioned FiLM layers embed dual watermarks through complementary band modulation.}
%   \label{fig:generator_encoder}
% \end{figure}

% %\subsection{Deployment Model}
% %\label{sec:deployment}
% %AudioAuth follows a \emph{proactive watermarking} paradigm~\cite{audioseal,c2pa2024}: the watermark is embedded during speech synthesis, directly within the TTS or voice conversion pipeline operated by an authorized provider, so that every generated utterance carries a verifiable provenance signature from the point of creation. This generation-time integration aligns with the Coalition for Content Provenance and Authenticity (C2PA) framework, which specifies soft-binding watermarks as a mechanism for linking audio content to its origin metadata~\cite{c2pa2024}. Specifically, the \emph{embedding side} (TTS providers, voice synthesis platforms) integrates AudioAuth's generator into their speech production pipeline, embedding provenance at generation time. The \emph{verification side} (biometric authentication systems, content platforms, forensic analysts) deploys only the detector and locator networks, requiring no shared secrets or key exchange.

% %At verification time, the detector and locator operate independently of the embedder, enabling three practical use cases: (i)~a voice biometric authentication system can query the watermark to confirm whether incoming speech originates from a registered synthesis source~\cite{das2020assessing,asvspoof2019}, (ii)~a content moderation platform can flag unlabeled synthetic audio for further review, and (iii)~a forensic investigator can extract the model-identifier bits to trace a spoofing attack back to the specific TTS system that produced it. Because detection requires only the trained extraction network and no secret key exchange, the deployment model naturally separates the trusted embedding side (model developers, content platforms) from the open verification side (biometric security systems, forensic analysts, and content moderators).

% %\noindent \textbf{Model Footprint and Inference Latency.}
% %The complete AudioAuth pipeline comprises 14.03M parameters (Generator 9.59M, Detector 4.31M, Locator 0.13M), with a 0.47M-parameter discriminator used only during training. At inference on a single GPU, watermark embedding requires 0.12\,s and detection 0.08\,s per one second of audio, enabling real-time deployment in authentication pipelines at ${\leq}200$\,ms total latency.

% \subsection{System Architecture}
% AudioAuth implements an end-to-end pipeline for robust audio watermarking through a three-stage hierarchical architecture illustrated in Figure~1. The system operates through three interconnected stages:
% \begin{enumerate}[nosep,leftmargin=*]
%   \item A \textbf{FiLM-based generator} that embeds dual watermarks (model identification and data authentication bits) into audio through frequency-partitioned multi-band modulation across hierarchical encoder layers, leveraging a SEANet-based encoder-decoder architecture~\cite{seanet} for multi-scale feature extraction.
%   \item An \textbf{augmentation stage} that applies temporal and audio effect transformations during training to enhance robustness against real-world attacks.
%   \item \textbf{Dual extraction networks}, comprising a detector for bit-level message recovery and a locator for temporal boundary identification, that jointly identify and localize watermarked regions.
% \end{enumerate}

% This hierarchical multi-scale architecture distributes watermark information across multiple frequency bands (\(B=4\)). This partition count is the minimum that supports the complementary even/odd scheme (2~even + 2~odd bands), providing sufficient spectral granularity to exploit the 70/30 weighting while maintaining low memory footprint and training stability; fewer bands ($B{=}2$) would collapse each watermark channel to a single band with no cross-band redundancy, while more bands ($B{=}8$) increase parameter count and training time without measurable MIoU improvement (see Supplementary Materials). The following subsections detail each component and its theoretical foundations.

% \subsection{Watermark Embedding and Modulation}

% Previous audio watermarking approaches, including signal-level perturbation methods such as AudioSeal and WavMark and representation-level approaches like DiscreteWM and Timbre Watermarking, typically embed watermarks via uniform or direct linear extensions of message-derived vectors. While effective for coarse watermark detection, these methods generally lack resilience to temporal modifications and spectral attacks, and often yield uneven distribution of watermark information across time and frequency. To address these limitations, AudioAuth introduces a hierarchical modulation architecture operating at \emph{multiple temporal and frequency scales}, as illustrated in Figure~1(a) (Supplementary~I).

% \begin{figure}[t]
% \centering
% \includegraphics[width=0.75\linewidth]{IEEEtran/figures/dual_embedding.png}
% \caption{Frequency-partitioned dual watermark embedding. The 32-bit message is split into model identifier (bits 0--15) and data authentication (bits 16--31) streams, processed through independent MLPs, and distributed across frequency bands with complementary 70/30 weighting to ensure resilience against frequency-selective attacks.}
% \label{fig:dual_watermark_embedding}
% \end{figure}

% The watermark embedding process begins by transforming an $n$-bit message into adaptive modulation parameters:

% \noindent \textbf{Message Processing.}
% A message processing module uses separate two-layer MLPs (Linear$\to$ReLU$\to$Linear, 16-bit $\to$ 64-dim) to map watermark bits to FiLM modulation parameters (scale $\gamma$ and shift $\beta$), enabling dynamic control over embedding strength. The 32-bit watermark is split into two channels: (1) model identification bits (first 16 bits), following a fixed alternating pattern $[0,1,\ldots,0,1]$ for source attribution, and (2) data watermark bits (last 16 bits) for authentication and content verification. As shown in Figure~3, the channels are embedded with complementary weighting: model identification bits receive 70\% strength in even frequency bands (0, 2) and 30\% in odd bands (1, 3), while data authentication bits invert this allocation. This design ensures that frequency-selective attacks impair at most one channel while preserving the other, providing robust path diversity (Supplementary~I).

% \noindent \textbf{Layer-wise FiLM Conditioning.}
% Each watermark channel is processed by an independent embedding multi-layer perceptron (MLP) that generates 64-dimensional FiLM parameters ($\gamma$, $\beta$) that match the base encoder channel width, at every encoder layer, as shown in Figure~2. Layer-wise FiLM conditioning enables adaptive control of watermark integration strength and frequency-band emphasis across the encoder hierarchy.
% Critically, this multi-scale integration distributes watermark information across four encoder layers with progressive downsampling ratios $[8, 5, 4, 2]$ (cumulative $320\times$), ensuring that coarse-level embeddings at higher layers survive global temporal modifications such as time-stretching, while fine-level embeddings at lower layers resist local perturbations such as frame-level jitter (Supplementary~I).

% \subsection{Augmentation Stage}
% To enhance watermark robustness against a wide range of potential modifications, we employ a two-level augmentation strategy during the training stage, as depicted in Figure~1(b), explained as follows:

% \subsubsection{Temporal Augmentations}
% \label{sec:temporal_structural_augmentation_combined}
% At this stage, we add two kinds of temporal augmentations, namely, segment-level transformations targeting localized regions and sequence-level transformations altering the entire temporal structure.

% For segment-level temporal augmentations, the framework operates on fixed-duration audio segments (0.1s) and modifies 20\% of randomly selected segments, applying with equal probability one of three transformations: replacing watermarked segments with non-watermarked counterparts, setting segments to silence, or replacing segments with audio from a different source.

% Complementing these, we implement sequence-level augmentations by randomly applying one transformation to the entire audio signal while preserving watermark content: reversing the temporal order, rotating by a random offset, or shuffling fixed-length segments (e.g., 0.5s). By forcing the model to identify watermarks across varied sequential patterns, it learns intrinsic features rather than positional cues, enabling robust sample-level detection even under significant reordering.

% \subsubsection{Audio Effect Augmentation}
% The second class of augmentation targets robustness against audio editing. To simulate real-world modifications, our augmentation pipeline applies a diverse set of audio effects, including high-pass, low-pass, and band-pass filtering; resampling; speed modification; random noise; audio boost and ducking; echo; pink noise; white Gaussian noise; smoothing; AAC and MP3 compression; and EnCodec compression. Collectively, these effects cover frequency filtering, temporal distortions, amplitude variations, noise injection, and lossy compression.

% Rather than using fixed augmentation parameters or static sampling probabilities, our approach dynamically adjusts augmentation strength and selects transformations according to the model’s current error profile, defined as a weighted combination of Bit Error Rate (BER) and the complement of Mean Intersection over Union (MIoU). This \emph{Dynamic Effect Scheduler}, inspired by curriculum learning principles~\cite{Bengio2009Curriculum}, prioritizes transformations that pose the greatest challenge during training.
% \begin{itemize}
% \item \textbf{Bit Error Rate (BER)}: The fraction of incorrectly decoded watermark bits, directly measuring recovery accuracy (lower is better).
% \item \textbf{Mean Intersection over Union (MIoU)}: Measures the overlap between predicted and true watermarked regions, quantifying localization precision (higher is better).
% \end{itemize}
% Specifically, the scheduler adaptively controls both the selection probability and effect parameters (e.g., filter cutoffs and noise levels) based on real-time BER and MIoU computed on augmented samples. By emphasizing transformations associated with higher BER or lower MIoU, the scheduler progressively improves robustness to the most challenging distortions.

% Algorithm~1 details the Dynamic Effect Scheduler, which updates effect probabilities via a temperature-scaled softmax:
% \begin{equation}
% \resizebox{0.9\linewidth}{!}{$
% p_{t+1}(e) = \text{softmax}\left(\frac{w_1 \cdot \text{BER}_{\text{ema}}(e) + w_2 \cdot (1 - \text{MIoU}_{\text{ema}}(e))}{T}\right)
% \label{eq:dynamic_scheduler_prob_update_revised}
% $}
% \end{equation}
% Weights $w_1{=}0.8$ and $w_2{=}0.2$ were selected via grid search over $\{0.5/0.5,\; 0.7/0.3,\; 0.8/0.2,\; 0.9/0.1\}$ on the validation set, prioritizing detection (BER) over localization (MIoU); the $0.8/0.2$ split yielded the lowest combined error, as equal weighting under-penalized high-BER effects while $0.9/0.1$ neglected localization degradation. The temperature $T$ anneals from $1.0$ to $0.7$, selected via grid search over $\{0.5, 0.7, 0.9, 1.0\}$ on the validation set to balance exploration of underrepresented effects against exploitation of the hardest distortions ($T{=}0.5$ over-concentrates on a single effect, while $T{\geq}0.9$ underweights difficult distortions). Exponential Moving Average (EMA) smoothing ($\beta{=}0.9$, following standard practice for stable adaptation~\cite{Bengio2009Curriculum}) ensures smooth tracking of the most challenging distortions.

% The Dynamic Effect Scheduler adaptively adjusts per-effect sampling probabilities using EMA-smoothed BER and MIoU metrics, progressively narrowing augmentation parameters toward harder settings as the model improves. All augmentations are applied on-the-fly during training. Full algorithmic details are provided in Supplementary Section~IV.
% \subsection{Dual-Network Architecture for Watermark Detection and Localization}

% The architecture employs two specialized neural networks with shared design principles but distinct objectives, as shown in Figure~1(c): (1) a \textit{Detector Network} for robust message recovery via multi-scale feature analysis using an encoder–decoder architecture (Figure~1, Right), and (2) a \textit{Locator Network} for precise temporal identification of watermarked regions through streamlined high-resolution processing. This bifurcated design offers three key advantages. First, it decouples orthogonal tasks—bit-level extraction and boundary localization—allowing each network to optimize independently without performance trade-offs. Second, the Detector’s deeper 4-layer encoder preserves multi-scale features critical for accurate bit recovery under severe distortions, while the Locator’s streamlined 2-layer architecture enables real-time sample-level precision (±2 samples at 16~kHz) with minimal overhead (0.13M parameters). Third, mask-pooled temporal aggregation using the Locator's confidence scores guides the Detector's bit extraction, providing robustness to temporal desynchronization and fragmented watermark scenarios, and maintaining MIoU~$>$~0.98 even with 90% watermark removal~\cite{cox2002digital}.

% \indent This dual-network strategy addresses core limitations of single-stream architectures, which struggle to balance temporal precision with robust feature extraction under compound attacks. By employing task-specific encoder designs—a full 4-layer SEANet encoder for the Detector~\cite{hilcodec} and a streamlined 2-layer variant for the Locator—the architecture aligns with established signal processing principles~\cite{cox2002digital}. The Locator further maintains strict input-output temporal alignment through symmetric padding and stride management, enabling detection windows as small as 50ms with positional accuracy of $\pm$2 samples at 16\,kHz (Supplementary~II). Complete architectural details, including layer specifications, parameter counts, and temporal alignment formulations, are provided in the Supplementary Materials, Section~II.
% \subsection{Loss Functions}
% AudioAuth combines reconstruction, detection, localization, and adversarial losses into a multi-objective function optimized through hierarchical weighting.

% \noindent \textbf{Reconstruction Loss.} Building on spectral reconstruction losses from neural codecs~\cite{codecwithimprovedrvqgan}, we introduce a weighted scheme prioritizing time-domain fidelity:
% \begin{equation}
% \resizebox{\linewidth}{!}{$
% \mathcal{L}_{\text{rec}} = \lambda_{\text{wave}}\mathcal{L}_{\text{wave}} + \lambda_{\text{spec}}\mathcal{L}_{\text{spec}} + \lambda_{\text{mel}}\mathcal{L}_{\text{mel}}
% $}
% \end{equation}
% where $\mathcal{L}_{\text{wave}} = |x-\hat{x}|_1$ represents direct waveform matching between original ($x$) and watermarked ($\hat{x}$) audio, $\mathcal{L}_{\text{spec}}$ computes multi-scale Short-Time Fourier Transform (STFT) loss (window sizes 512/2048), and $\mathcal{L}_{\text{mel}}$ calculates mel-band (150/80 filter banks) spectral loss. We set $\lambda_{\text{wave}}{=}1.0$, $\lambda_{\text{spec}}{=}0.5$, $\lambda_{\text{mel}}{=}0.3$, empirically tuned to emphasize temporal accuracy over spectral fidelity.

% During training, ground truth labels are derived from the embedding pipeline: the binary mask $m_i{=}1$ for sample positions retaining the original watermarked signal and $m_i{=}0$ for positions replaced by silence, unwatermarked, or alternate-source audio during augmentation (Section~\ref{sec:temporal_structural_augmentation_combined}), and $y_i \in \{0,1\}$ denotes the original watermark bits produced by the generator.

% \noindent \textbf{Localization Loss.} To detect watermark boundaries, we employ:
% \begin{equation}
% \resizebox{\linewidth}{!}{$
% \mathcal{L}_{\text{loc}} = -\frac{1}{N}\sum_{i=1}^N\left[m_i\log p_i^{\text{(loc)}} + (1-m_i)\log(1-p_i^{\text{(loc)}})\right]
% $}
% \end{equation}
% where $N$ is the number of time-domain samples per audio segment, $p_i^{\text{(loc)}}$ denotes the locator network's prediction of watermark presence at position $i$, and $m_i$ is the ground truth mask defined above. This loss optimizes temporal boundary detection to achieve precise localization of watermarked regions.

% \noindent \textbf{Detection Loss.} For detection, we jointly optimize two complementary BCE losses. The presence-masked detection loss focuses learning on watermarked regions using binary mask $m_i \in \{0,1\}$:
% \begin{equation}
% \resizebox{\linewidth}{!}{$
% \mathcal{L}_{\text{det}} = -\frac{1}{N}\sum_{i=1}^N m_i\left[y_i\log p_i^{\text{(det)}} + (1-y_i)\log(1-p_i^{\text{(det)}})\right]
% $}
% \end{equation}
% where $p_i^{\text{(det)}}$ represents the detector network's predicted probability of watermark bit value at position $i$, and $y_i \in \{0,1\}$ denotes ground truth watermark bits. This formulation concentrates gradient updates on areas containing actual watermark content, optimizing bit value accuracy rather than temporal boundaries.

% \noindent \textbf{Adversarial Loss.} Adversarial training employs $L{=}3$ multi-scale discriminators with weighted objectives:
% \begin{equation}
% \resizebox{\linewidth}{!}{$
% \mathcal{L}_{\text{adv}} = \lambda_{\text{gen}}\mathbb{E}\left[(1-D(G(x)))^2\right] + \lambda_{\text{feat}}\sum_{l=1}^L|f_l(x)-f_l(G(x))|_1
% $}
% \end{equation}
% where $f_l(x)$ represents feature maps at layer $l$ of discriminator $D$ when processing real audio $x$, and $f_l(G(x))$ corresponds to audio features of the generated audio. We use $\lambda_{\text{gen}}{=}1.0$ and $\lambda_{\text{feat}}{=}100.0$~\cite{codecwithimprovedrvqgan}. The feature matching term stabilizes training by preserving intermediate representations across $L{=}3$ discriminator scales.

% \noindent \textbf{Combined Loss.} The complete objective function integrates these components through task-specific weights:
% \begin{equation}
% \resizebox{\linewidth}{!}{$
% \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + \lambda_{\text{det}}\mathbb{E}_\epsilon[\mathcal{L}_{\text{det}}^\epsilon] + \lambda_{\text{loc}}\mathbb{E}_\epsilon[\mathcal{L}_{\text{loc}}^\epsilon] + \mathcal{L}_{\text{adv}}
% $}
% \end{equation}
% where the expectation $\mathbb{E}_\epsilon$ computes the average over augmentation variants from our dynamic scheduler, implemented through parallel processing of $K{=}3$ augmented copies per sample. We set $\lambda_{\text{det}}{=}10.0$ and $\lambda_{\text{loc}}{=}1.0$, prioritizing detection robustness (BER) over localization precision (MIoU).

\section{Proposed Method}\label{sec:method}



\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth, height=12cm, keepaspectratio]{IEEEtran/figures/generator_encoder.png}
  \caption{Generator encoder architecture. Frequency-partitioned FiLM layers embed dual watermarks through complementary band modulation.}
  \label{fig:generator_encoder}
\end{figure}

AudioAuth implements an end-to-end pipeline for robust audio watermarking through a sophisticated hierarchical architecture illustrated in Figure~1. The system operates through three interconnected stages: (1) a \textbf{FiLM-based generator} that embeds dual watermarks (source-attribution and integrity-verification bits) into audio through frequency-partitioned multi-band modulation across hierarchical encoder layers, leveraging a SEANet-based encoder-decoder architecture~\cite{seanet} for hierarchical multi-scale feature extraction; (2) a \textbf{augmentation stage} that applies temporal and audio effect transformations during training to enhance robustness against real-world attacks, and (3) \textbf{dual extraction networks}---a detector for bit-level message recovery and a locator for temporal boundary identification---that jointly identify and localize watermarked regions. This hierarchical multi-scale architecture distributes watermark information across multiple frequency bands (typically \(B=4\)) to balance frequency resolution and computational efficiency, as this partition count provides sufficient granularity to exploit the complementary 70/30 weighting scheme across even and odd bands while maintaining low memory footprint and training stability. The following subsections detail each component and its theoretical foundations.

%\textbf{Deployment Scenario.} AudioAuth operates in a \emph{proactive watermarking} setting: watermarks are embedded at content generation time within a trusted synthesis pipeline (e.g., by the TTS model provider or voice synthesis service), before distribution. This integration point ensures that all generated audio carries verifiable provenance from its origin. Detection and localization occur post-distribution when verifying authenticity or investigating potential manipulation—for instance, when a speaker verification system encounters audio of uncertain provenance, or when forensic analysis is required to determine if speech has been synthetically generated or tampered with. This deployment model assumes that the embedding process is controlled by authorized parties (model developers, content platforms), while detection may be performed by any verifier with access to the detector, including biometric security systems, content moderation platforms, and forensic investigators.

%\textbf{Deployment Scenario.} AudioAuth follows a \emph{proactive watermarking} paradigm~\cite{audioseal,c2pa2024}: the watermark is embedded during speech synthesis, directly within the TTS or voice conversion pipeline operated by an authorized provider, so that every generated utterance carries a verifiable provenance signature from the point of creation. This generation-time integration aligns with the Coalition for Content Provenance and Authenticity (C2PA) framework, which specifies soft-binding watermarks as a mechanism for linking audio content to its origin metadata~\cite{c2pa2024}.

%At verification time, the detector and locator operate independently of the embedder, enabling three practical use cases: (i)~a voice biometric authentication system can query the watermark to confirm whether incoming speech originates from a registered synthesis source~\cite{das2020assessing,asvspoof2019}, (ii)~a content moderation platform can flag unlabeled synthetic audio for further review, and (iii)~a forensic investigator can extract the model-identifier bits to trace a spoofing attack back to the specific TTS system that produced it. Because detection requires only the trained extraction network and no secret key exchange, the deployment model naturally separates the trusted embedding side (model developers, content platforms) from the open verification side (biometric security systems, forensic analysts, and content moderators).

\vspace{-10pt}
\subsection{Watermark Embedding and Modulation}

Previous audio watermarking approaches, including signal-level perturbation methods such as AudioSeal and WavMark and representation-level approaches like DiscreteWM and Timbre Watermarking, typically embed watermarks via uniform or direct linear extensions of message-derived vectors. While effective for coarse watermark detection, these methods generally lack resilience to temporal modifications and spectral attacks, and often yield uneven distribution of watermark information across time and frequency. To address these limitations, AudioAuth introduces a hierarchical modulation architecture operating at \emph{multiple temporal and frequency scales}, as illustrated in Figure~1(a) (see Supplementary Section~I for full specifications).

\begin{figure}[t]
\centering
\includegraphics[width=0.75\linewidth]{IEEEtran/figures/dual_embedding.png}
\caption{Frequency-partitioned dual watermark embedding. The 32-bit message is split into source-attribution identifier (bits 0--15) and integrity-verification (bits 16--31) streams, processed through independent MLPs, and distributed across frequency bands with complementary 70/30 weighting to ensure resilience against frequency-selective attacks.}
\label{fig:dual_watermark_embedding}
\end{figure}

The watermark embedding process begins by transforming an $n$-bit message into adaptive modulation parameters:

\begin{itemize}
\item A message processing module uses separate multi-layer perceptrons (MLPs) to map watermark bits to FiLM modulation parameters (scale $\gamma$ and shift $\beta$), enabling dynamic control over embedding strength. The 32-bit watermark is split into two channels: (1) source-attribution bits (first 16 bits), following a fixed alternating pattern $[0,1,\ldots,0,1]$ for source attribution, and (2) integrity-verification bits (last 16 bits) for content-integrity verification. As shown in Figure~3, the channels are embedded with complementary weighting: 
source-attribution bits receive 70\% strength in even frequency bands (0, 2) and 30\% in odd bands (1, 3), while integrity-verification bits invert this allocation. This \textbf{70/30 weighting was validated as optimal through an ablation across four allocation strategies}---100/0, 70/30, 50/50, and 30/70---where it achieves the highest dual-channel robustness (MIoU 0.984), whereas the uniform 100/0 configuration causes catastrophic data-channel failure (BER 12.4\%) under frequency-selective attacks (Supplementary Section~II-D, Table~VIII). This complementary design ensures that such attacks impair at most one channel while preserving the other, providing robust path diversity (full modulation specifications in Supplementary Section~I). % model identification bits receive 70\% strength in even frequency bands (0, 2) and 30\% in odd bands (1, 3), while data authentication bits invert this allocation. This design ensures that frequency-selective attacks impair at most one channel while preserving the other, providing robust path diversity (details in Supplementary Materials, Section~I).
\item Each watermark channel is processed by an independent embedding MLP that generates 64-dimensional FiLM parameters ($\gamma$, $\beta$) at every encoder layer, as shown in Figure~2. Layer-wise FiLM conditioning enables adaptive control of watermark integration strength and frequency-band emphasis across the encoder hierarchy.
\end{itemize}
\vspace{-10pt}

\subsection{Augmentation Stage}
To enhance watermark robustness against a wide range of potential modifications, we employ a two-level augmentation strategy during the training stage, as depicted in Figure~1(b), explained as follows:

\subsubsection{Temporal Augmentations}
\label{sec:temporal_structural_augmentation_combined}
At this stage, we add two kinds of temporal augmentations, namely, segment-level transformations targeting localized regions and sequence-level transformations altering the entire temporal structure.

For segment-level temporal augmentations, the framework operates on fixed-duration audio segments (0.1s) and modifies 20\% of randomly selected segments, applying with equal probability one of three transformations: replacing watermarked segments with non-watermarked counterparts, setting segments to silence, or replacing segments with audio from a different source. Complementing these, we implement sequence-level augmentations by randomly applying one transformation to the entire audio signal while preserving watermark content: reversing the temporal order, rotating by a random offset, or shuffling fixed-length segments (e.g., 0.5s). By forcing the model to identify watermarks across varied sequential patterns, it learns intrinsic features rather than positional cues, enabling robust sample-level detection even under significant reordering.

\subsubsection{Audio Effect Augmentation}
The second class of augmentation targets robustness against audio editing. To simulate real-world modifications, our augmentation pipeline applies a diverse set of individual audio effects, including high-pass, low-pass, and band-pass filtering; resampling; speed modification; random noise; audio boost and ducking; echo; pink noise; white Gaussian noise; smoothing; AAC and MP3 compression; and EnCodec compression. Collectively, these effects cover frequency filtering, temporal distortions, amplitude variations, noise injection, and lossy compression. To address \textit{adaptive, multi-stage attacks}, the pipeline also composes compound distortion chains by sequentially applying two or more effects per training sample (e.g., codec compression followed by bandpass filtering, or noise injection followed by speed modification). 
The Dynamic Effect Scheduler (Algorithm~1) governs both individual effect selection and chain composition, prioritizing compound sequences that yield the highest detection difficulty based on real-time BER and MIoU feedback, thereby exposing the model to the adversary-aware distortion pipelines it will encounter at inference.
% The second class of augmentation targets robustness against audio editing. To simulate real-world modifications, our augmentation pipeline applies a diverse set of audio effects, including high-pass, low-pass, and band-pass filtering; resampling; speed modification; random noise; audio boost and ducking; echo; pink noise; white Gaussian noise; smoothing; AAC and MP3 compression; and EnCodec compression. Collectively, these effects cover frequency filtering, temporal distortions, amplitude variations, noise injection, and lossy compression.

Rather than using fixed augmentation parameters or static sampling probabilities, our approach dynamically adjusts augmentation strength and selects transformations according to the model’s current error profile, defined as a weighted combination of Bit Error Rate (BER) and the complement of Mean Intersection over Union (MIoU). This \emph{Dynamic Effect Scheduler}, inspired by curriculum learning principles~\cite{Bengio2009Curriculum}, prioritizes transformations that pose the greatest challenge during training. Here, \textbf{Bit Error Rate (BER)} denotes the fraction of incorrectly decoded watermark bits, directly measuring recovery accuracy (lower is better), while \textbf{Mean Intersection over Union (MIoU)} measures the overlap between predicted and true watermarked regions, quantifying localization precision (higher is better). Specifically, the scheduler adaptively controls both the selection probability and effect parameters (e.g., filter cutoffs and noise levels) based on real-time BER and MIoU computed on augmented samples. By emphasizing transformations associated with higher BER or lower MIoU, the scheduler progressively improves robustness to the most challenging distortions.

Algorithm~1 details the Dynamic Effect Scheduler, which updates effect probabilities via a temperature-scaled softmax:
\begin{equation}
\resizebox{0.9\linewidth}{!}{$
p_{t+1}(e) = \text{softmax}\left(\frac{w_1 \cdot \text{BER}_{\text{ema}}(e) + w_2 \cdot (1 - \text{MIoU}_{\text{ema}}(e))}{T}\right)
\label{eq:dynamic_scheduler_prob_update_revised}
$}
\end{equation}
We set $w_1=0.8$ and $w_2=0.2$ to prioritize detection (BER) over localization (MIoU). The temperature $T$ anneals from $1.0$ to $0.7$ to balance exploration with exploitation, while exponential moving averages ($\beta=0.9$) ensure stable adaptation to the most challenging distortions.


\begin{algorithm}[t]
\caption{Dynamic Effect Scheduling}
\small
\begin{algorithmic}
\REQUIRE Effects $E$, smoothing factor $\beta=0.9$
\STATE Initialize uniform probabilities $p_0(e) = 1/|E|$
\STATE Initialize $\text{BER}_{\text{ema}}(e)=0.5$, $\text{MIoU}_{\text{ema}}(e)=0.5$ for all $e \in E$
\FOR{each training iteration $t$}
\STATE Sample effects according to probabilities $p_t(e)$
\STATE Apply selected effects with parameters sampled from $P(\theta|e)$
\STATE Compute $\text{BER}_t(e)$, $\text{MIoU}_t(e)$ for each applied effect
\STATE Update EMAs:
\STATE $\text{BER}_{\text{ema}}(e) \gets \beta \cdot \text{BER}_{\text{ema}}(e) + (1-\beta) \cdot \text{BER}_t(e)$
\STATE $\text{MIoU}_{\text{ema}}(e) \gets \beta \cdot \text{MIoU}_{\text{ema}}(e) + (1-\beta) \cdot \text{MIoU}_t(e)$
\STATE Update probabilities using weighted performance metrics:
\STATE $p_{t+1}(e) = \text{softmax}\left(\frac{w_1 \cdot \text{BER}_{\text{ema}}(e) + w_2 \cdot (1 - \text{MIoU}_{\text{ema}}(e))}{T}\right)$
\STATE Update parameter distribution $P(\theta|e)$ based on success rates
\ENDFOR
\STATE \textbf{return} Model with best validation performance
\end{algorithmic}
\label{alg:scheduler_revised}
\end{algorithm}

Importantly, all augmentations are applied on-the-fly during training to each batch of audio samples, rather than pre-generating augmented datasets. This dynamic approach ensures diverse training conditions while maintaining memory efficiency, though it introduces approximately 15\% computational overhead compared to training without augmentations due to real-time audio processing operations.
\vspace{-3.5mm}
\subsection{Dual-Network Architecture for Watermark Detection and Localization}

The architecture employs two specialized neural networks with shared design principles but distinct objectives, as shown in Figure~1(c): (1) a \textit{Detector Network} for robust message recovery via multi-scale feature analysis using an encoder–decoder architecture (Figure~1, Right), and (2) a \textit{Locator Network} for precise temporal identification of watermarked regions through streamlined high-resolution processing. This bifurcated design offers three key advantages. First, it decouples orthogonal tasks—bit-level extraction and boundary localization—allowing each network to optimize independently without performance trade-offs. Second, the Detector’s deeper 4-layer encoder preserves multi-scale features critical for accurate bit recovery under severe distortions, while the Locator’s streamlined 2-layer architecture enables real-time sample-level precision (±2 samples at 16~kHz) with minimal overhead (0.13M parameters). Third, mask-pooled temporal aggregation using the Locator’s confidence scores guides the Detector’s bit extraction, providing robustness to temporal desynchronization and fragmented watermark scenarios, and maintaining MIoU~$>$~0.98 even with 90% watermark removal~\cite{cox2002digital}.

This dual-network strategy addresses core limitations of single-stream architectures, which struggle to balance temporal precision with robust feature extraction under compound attacks. By employing task-specific encoder designs—a full 4-layer SEANet encoder for the Detector~\cite{hilcodec} and a streamlined 2-layer variant for the Locator—the architecture aligns with established signal processing principles~\cite{cox2002digital}. Complete architectural details, including layer specifications, parameter counts, and temporal alignment formulations, are provided in Supplementary Section~I-B.
\vspace{-10pt}
\subsection{Loss Functions}
AudioAuth combines reconstruction, detection, localization, and adversarial losses into a multi-objective function optimized through hierarchical weighting.

\noindent \textbf{Reconstruction Loss.} Building on spectral reconstruction losses from neural codecs~\cite{codecwithimprovedrvqgan}, we introduce a weighted scheme prioritizing time-domain fidelity:
\begin{equation}
\resizebox{\linewidth}{!}{$
\mathcal{L}_{\text{rec}} = \lambda_{\text{wave}}\mathcal{L}_{\text{wave}} + \lambda_{\text{spec}}\mathcal{L}_{\text{spec}} + \lambda_{\text{mel}}\mathcal{L}_{\text{mel}}
$}
\end{equation}
where $\mathcal{L}_{\text{wave}} = |x-\hat{x}|_1$ represents direct waveform matching between original ($x$) and watermarked ($\hat{x}$) audio, $\mathcal{L}_{\text{spec}}$ computes multi-scale STFT loss (window sizes 512/2048), and $\mathcal{L}_{\text{mel}}$ calculates mel-band (150/80 filter banks) spectral loss. Our weighting scheme emphasizes temporal accuracy over spectral fidelity based on ablation studies.

\noindent \textbf{Localization Loss.} To detect watermark boundaries, we employ:
\begin{equation}
\resizebox{\linewidth}{!}{$
\mathcal{L}_{\text{loc}} = -\frac{1}{N}\sum_{i=1}^N\left[m_i\log p_i^{\text{(loc)}} + (1-m_i)\log(1-p_i^{\text{(loc)}})\right]
$}
\end{equation}
where $p_i^{\text{(loc)}}$ denotes the locator network's prediction of watermark presence at position $i$, while $m_i$ serves as the ground truth mask. This loss optimizes temporal boundary detection to achieve precise localization of watermarked regions.

\noindent \textbf{Detection Loss.} For detection, we jointly optimize two complementary BCE losses. The presence-masked detection loss focuses learning on watermarked regions using binary mask $m_i \in \{0,1\}$:
\begin{equation}
\resizebox{\linewidth}{!}{$
\mathcal{L}_{\text{det}} = -\frac{1}{N}\sum_{i=1}^N m_i\left[y_i\log p_i^{\text{(det)}} + (1-y_i)\log(1-p_i^{\text{(det)}})\right]
$}
\end{equation}
where $p_i^{\text{(det)}}$ represents the detector network's predicted probability of watermark bit value at position $i$, and $y_i \in \{0,1\}$ denotes ground truth watermark bits. This formulation concentrates gradient updates on areas containing actual watermark content, optimizing bit value accuracy rather than temporal boundaries.

\noindent \textbf{Adversarial Loss.} Adversarial training employs multi-scale discriminators with weighted objectives:
\begin{equation}
\resizebox{\linewidth}{!}{$
\mathcal{L}_{\text{adv}} = \lambda_{\text{gen}}\mathbb{E}\left[(1-D(G(x)))^2\right] + \lambda_{\text{feat}}\sum_{l=1}^L|f_l(x)-f_l(G(x))|_1
$}
\end{equation}
where $f_l(x)$ represents feature maps at layer $l$ of discriminator $D$ when processing real audio $x$, and $f_l(G(x))$ corresponds to audio features of the generated audio. The feature matching term stabilizes training by preserving intermediate representations across multiple discriminator layers ($L$).

\noindent \textbf{Combined Loss.} The complete objective function integrates these components through task-specific weights:
\begin{equation}
\resizebox{\linewidth}{!}{$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + \lambda_{\text{det}}\mathbb{E}_\epsilon[\mathcal{L}_{\text{det}}^\epsilon] + \lambda_{\text{loc}}\mathbb{E}_\epsilon[\mathcal{L}_{\text{loc}}^\epsilon] + \mathcal{L}_{\text{adv}}
$}
\end{equation}
where the expectation $\mathbb{E}_\epsilon$ computes the average over augmentation variants from our dynamic scheduler, implemented through parallel processing of augmented copies per sample. This weighting prioritizes detection robustness (BER) over localization precision (MIoU).

\section{Experiments}\label{sec:experiment}
\subsection{Dataset}
To achieve domain generalization in speech processing, we utilize multiple diverse voice datasets~\cite{librispeech, commonvoice, cmuarctic, dipco}, all resampled to 16\,kHz, for AudioAuth model training. We implement stratified batch sampling with controlled distribution as follows. LibriSpeech~\cite{librispeech} contributes 40\% of training data, comprising 1{,}000 hours of read English speech (${\sim}$200{,}000 training segments at 1.0\,s), with a held-out set of 100 unseen speakers reserved for testing. Common Voice~\cite{commonvoice} accounts for 30\%, providing 200 hours spanning 10 languages (${\sim}$150{,}000 training segments), with 5{,}000 test clips from 1{,}000 new speakers held out for testing. CMU ARCTIC~\cite{cmuarctic} supplies 15\% with 20 hours of professional speech (${\sim}$75{,}000 training segments), tested using a leave-two-speakers-out strategy. The remaining 15\% comes from DiPCo~\cite{dipco}, offering 40 hours of conversational speech (${\sim}$75{,}000 training segments), with the last 10 sessions reserved for testing.

Furthermore, we assessed cross-domain performance on two entirely unseen datasets reserved exclusively for testing. RAVDESS~\cite{ravdess} contains 7,356 audio files from 24 professional actors expressing 8 emotions (neutral, calm, happy, sad, angry, fearful, surprise, disgust) in both speech and singing modalities. ASVspoof 2019~\cite{asvspoof2019} includes two evaluation tracks: Logical Access (LA) with 2,548 bona-fide and 22,296 spoofed (TTS/VC) utterances, and Physical Access (PA) with 5,400 bona-fide and 24,300 spoofed replay utterances under diverse acoustic conditions; we use this dataset to assess our watermarking model’s robustness to unseen replay and deepfake attacks.

\subsection{Experimental Setup and Training Strategy}

The experiments were conducted using the PyTorch framework on NVIDIA Quadro RTX~8000 GPUs. All audio samples were resampled to 16~kHz. The training set comprised 500{,}000 audio segments of 1.0~s duration, while validation was performed on 50 held-out segments. The validation set served exclusively for convergence monitoring (BER $\leq$ 0.1\%, MIoU $\geq$ 98\%) and stage-transition decisions rather than hyperparameter selection; all hyperparameters were determined via grid search on separate held-out data (Section~III). For comprehensive evaluation, we employed 1{,}000 extended audio clips of 10.0~s duration from entirely unseen RAVDESS and ASVspoof datasets.

AudioAuth adopts a three-stage progressive training strategy that moves from simple to increasingly challenging conditions to balance two competing objectives: robust watermark embedding and high perceptual audio fidelity. Direct end-to-end training under all distortions, as adopted by existing audio watermarking techniques~\cite{audioseal,wavmark}, often biases learning toward robustness too early, producing audible artifacts that violate imperceptibility. By first learning clean embedding and then gradually introducing distortions, the model develops stable imperceptible representations before confronting aggressive perturbations, following curriculum-learning principles~\cite{Bengio2009Curriculum}.

All stages use the AdamP optimizer~\cite{Heo2021AdamP} with a learning rate of $5\times10^{-4}$, $\beta_{1}=0.8$, $\beta_{2}=0.99$, $\epsilon=10^{-8}$, and zero weight decay, following a constant schedule with a minimum learning rate of $1\times10^{-5}$. Gradient clipping stabilizes training, with norms capped at $1000.0$ for the watermarking model and $10.0$ for the discriminator.

\textbf{Stage~1: Identity Mapping and Foundation Learning.}
In the first stage, the model learns fundamental watermark embedding and reconstruction using only identity transformations, with no audio effects applied. This establishes the baseline encoder--decoder mapping and enables imperceptible watermark integration without robustness-oriented augmentations. Training progress is tracked using an exponential moving average (EMA) with $\beta=0.9$, and convergence is defined by achieving a bit error rate (BER) below 10\% and mean Intersection-over-Union (MIoU) above 80\%.

\textbf{Stage~2: Robustness Training with Adaptive Augmentations.}
Stage~2 introduces comprehensive audio augmentations to build robustness once imperceptible embedding is established. Six audio effects with adaptive parameters are enabled: white noise (15--30~dB), pink noise (20--35~dB), low-pass filtering (3--6~kHz), high-pass filtering (0.5--2~kHz), band-pass filtering (seven telephony--wideband bands), and volume scaling (0.5--1.1), in addition to identity mapping. Temporal augmentations include segment-level edits, where 20\% of 0.1~s segments are replaced with silence or alternate content, as well as sequence-level operations such as reversal, rotation, and segment shuffling.

An adaptive scheduler selects the most challenging effects using EMA-based tracking ($\beta=0.9$) with stricter convergence targets of BER~$\leq$~0.1\% and MIoU~$\geq$~98\%. To prevent catastrophic forgetting, progression to the final stage requires BER to remain below 10\%.

\textbf{Stage~3: Fidelity Refinement.}
In the final stage, audio reconstruction loss weights are increased to prioritize perceptual quality while preserving the robustness achieved in Stage~2. This refinement corrects residual artifacts introduced during aggressive robustness training and ensures that detection performance does not come at the expense of audio naturalness.

Training was performed for a total of 600{,}000 iterations with a batch size of 32 on NVIDIA Quadro RTX~8000 GPUs, with stage transitions determined by convergence criteria (BER and MIoU thresholds) rather than fixed iteration counts. Validation was conducted every 1{,}000 iterations to monitor convergence across stages. \\
\noindent \textbf{Model Footprint and Inference Latency.}
The complete AudioAuth pipeline comprises $14.03$M parameters (Generator $9.59$M, Detector $4.31$M, Locator $0.13$M), with a $0.47$M-parameter discriminator used only during training. At inference on a single NVIDIA Quadro RTX~8000 GPU, watermark embedding requires 0.12\,s and detection 0.08\,s per one second of audio, enabling real-time deployment in identity authentication pipelines at ${\leq}200$\,ms total latency. These timings are \textit{hardware-dependent}; the reported figures represent a baseline single-GPU configuration without inference-time optimizations such as TensorRT or mixed-precision acceleration, and actual latency will \emph{vary with available compute} resources.

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{IEEEtran/figures/spectrogram_grid.png}
  \caption{Spectrogram comparison across watermarking methods on a RAVDESS utterance (STFT: 2048 window, 512 hop, Hann). \textbf{Top:} Reconstructed spectrograms (0--4096~Hz). \textbf{Bottom:} Spectral difference maps (watermarked minus original). AudioAuth exhibits minimal distortion concentrated in perceptually less sensitive high-frequency bands, consistent with its frequency-partitioned FiLM embedding.}
  \label{fig:spectrogram-comparison}
\end{figure*}

\subsection{Evaluation Metrics}
% We evaluate watermark performance using Bit Error Rate (BER) for message recovery accuracy and Mean Intersection over Union (MIoU) for localization precision~\cite{cox2002digital,audioseal,wavmark}. Perceptual quality is assessed using Perceptual Evaluation of Speech Quality (PESQ)~\cite{pesq}, Short-Time Objective Intelligibility (STOI)~\cite{stoi}, Signal-to-Interference Ratio (SISNR), and ViSQOL~\cite{visqol}.
We evaluate watermark performance using Bit Error Rate (BER) for message recovery accuracy and Mean Intersection over Union (MIoU) for localization precision~\cite{cox2002digital,audioseal,wavmark}. Watermark detection reliability is measured via True Positive Rate (TPR), False Positive Rate (FPR), and False Negative Rate (FNR). Perceptual quality is assessed using Perceptual Evaluation of Speech Quality (PESQ)~\cite{pesq}, Short-Time Objective Intelligibility (STOI)~\cite{stoi}, Signal-to-Interference Ratio (SISNR), and ViSQOL~\cite{visqol}. For source attribution experiments, we report classification accuracy as the fraction of samples correctly assigned to their generating TTS system. For speaker verification impact analysis, we adopt Equal Error Rate (EER) and minimum Detection Cost Function (minDCF, $C_\mathrm{miss}=C_\mathrm{fa}=1$, $P_\mathrm{target}=0.05$)~\cite{nagrani2020voxceleb}, which are standard biometric verification metrics quantifying the trade-off between false acceptance and false rejection.

\section{Results}\label{sec:results}

In order to provide a transparent and comprehensive evaluation, all experiments were repeated in triplicate. Results are reported as means across three independent runs; standard deviations were below 0.005 for all metrics across three runs and are omitted from the main tables for clarity. 

\vspace{-2.5mm}
\subsection{Audio Quality Assessment and Trade-Off Analysis}

Quantitative comparisons focus on signal-level methods providing sample-wise localization (AudioAuth, WaveVerify, AudioSeal, WavMark) and WMCodec as a representation-level baseline; remaining methods (DiscreteWM, Timbre, FakeMark, AudioMarkNet, SyncGuard) lack temporal boundary detection architectures required for the localization-centric evaluations in Tables~II--V.

Table~I presents detailed results for perceptual audio quality metrics (PESQ, STOI, ViSQOL, SISNR) on the unseen evaluation set of RAVDESS and ASVspoof $2019$, with corresponding spectrogram visualizations shown in Fig.~4. AudioAuth achieves the highest ViSQOL (4.76), competitive PESQ (4.42), strong STOI (0.96), and robust SISNR (29.26~dB). In comparison, WaveVerify attains the best STOI (1.00) with ViSQOL of 4.72, AudioSeal records competitive PESQ (4.39), WavMark demonstrates superior signal fidelity with SISNR of 36.28~dB, WMCodec achieves PESQ of 3.35 at 6~kbps reflecting codec-efficiency constraints, and Timbre prioritizes anti-cloning robustness with lower PESQ (1.15) and STOI (0.87). The modest STOI and SISNR gap relative to WaveVerify and WavMark reflects a deliberate design trade-off: AudioAuth's dual-channel encoding distributes watermark energy across both even and odd frequency bands to carry attribution and integrity payloads simultaneously, whereas single-channel methods concentrate their payload in fewer spectral regions with lower per-band energy; this broader spectral footprint marginally reduces STOI and SISNR but enables the joint attribution--integrity capability that single-channel designs cannot provide. 

The spectral difference maps in Fig.~4 (bottom row) visually corroborate these quantitative findings: AudioAuth introduces minimal artifacts concentrated in perceptually less sensitive high-frequency bands, whereas methods like WMCodec and Timbre exhibit more pronounced spectral distortions across broader frequency ranges. AudioAuth's comprehensive performance demonstrates a well-balanced design that excels in perceptual quality while maintaining effective watermark robustness across diverse quality metrics. These quality metrics confirm that AudioAuth's watermark embedding preserves the acoustic features critical for downstream speaker verification pipelines, where even minor spectral distortions can degrade enrollment--verification cosine similarity and elevate equal error rates~\cite{Desplanques_2020}.
\begin{table}[htbp!]
  \caption{Comprehensive Audio Quality Comparison across Audio Watermarking Methods. Results are aggregated across RAVDESS (7,356 clips) and ASVspoof 2019 LA/PA evaluation sets (54,544 clips), reporting mean values across both datasets to assess cross-domain generalization. Bold values indicate the best performance for each metric.}
  \label{tab:audio_quality}
  \centering
  \small 
  \renewcommand{\arraystretch}{1.2}
  \setlength{\tabcolsep}{5pt}
  \begin{tabular}{@{}lcccc@{}}
    \toprule
    Method & PESQ & STOI & ViSQOL & SISNR (dB) \\
    \midrule
    AudioAuth (Ours) & \textbf{4.42} & 0.96 & \textbf{4.76} & 29.26 \\
    WaveVerify & 4.34 & \textbf{1.00} & 4.72 & 24.23 \\
    AudioSeal & 4.39 & 0.99 & 4.63 & 25.24 \\
    WavMark & \textbf{4.42} & 0.98 & 4.64 & \textbf{36.28} \\
    WMCodec & 3.35 & 0.95 & 4.10 & 24.00 \\
    Timbre & 1.15 & 0.87 & 3.70 & 21.50 \\
    \bottomrule
  \end{tabular}
\end{table}

\vspace{-2.5mm}
\subsection{Quantitative Evaluation under Diverse Audio Effects}

To evaluate AudioAuth's robustness, we compared it against AudioSeal~\cite{audioseal}, WavMark~\cite{wavmark}, WaveVerify~\cite{waveverify2025}, and WMCodec~\cite{wmcodec2025} on the RAVDESS~\cite{ravdess} and ASVspoof 2019~\cite{asvspoof2019} datasets (Table~II). AudioAuth achieves near-perfect watermark detection (True Positive Rate (TPR) $\geq$ $0.995$, False Positive Rate (FPR) $\leq$ $0.005$) and strong localization for content-integrity verification (MIoU $\geq$ $0.976$) across all fourteen audio effects. Notably, under high-pass filtering (1500Hz), AudioAuth (MIoU $0.983$) significantly outperforms AudioSeal ($0.612$) and WavMark ($0.595$), while matching the resilience of WaveVerify ($0.981$) and WMCodec ($0.991$ accuracy). This confirms AudioAuth's advantage over baselines that degrade under frequency-selective attacks.

\begin{table*}[htbp!]
    \caption{Comprehensive Robustness Evaluation across 14 Audio Effects. Detection (TPR/FPR) and Localization (MIoU) evaluated on 1,000 clips (500 RAVDESS, 500 ASVspoof 2019) per effect, aggregated across both datasets. Bold = best per effect.}
  \label{tab:comprehensive-robustness-evaluation}
  \centering
  \small
  \renewcommand{\arraystretch}{1.25}
  \setlength{\tabcolsep}{2.5pt}
  \resizebox{\textwidth}{!}{%
    \begin{tabular}{l|cc|cc|cc|cc|c}
      \toprule
      \multirow{2}{*}{\textbf{Audio Effect (Eval.)}} & \multicolumn{2}{c|}{\textbf{AudioAuth (Ours)}} & \multicolumn{2}{c|}{\textbf{WaveVerify}} & \multicolumn{2}{c|}{\textbf{AudioSeal}} & \multicolumn{2}{c|}{\textbf{WavMark}} & \textbf{WMCodec} \\
      \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-10}
      & \textbf{Det. (TPR/FPR)} & \textbf{MIoU} & \textbf{Det. (TPR/FPR)} & \textbf{MIoU} & \textbf{Det. (TPR/FPR)} & \textbf{MIoU} & \textbf{Det. (TPR/FPR)} & \textbf{MIoU} & \textbf{Extr. Acc.}$^*$ \\
      \midrule
      Bandpass (500–5000Hz) & \textbf{0.999 (0.999/0.001)} & \textbf{0.981} & 0.998 (0.998/0.002) & 0.979 & 0.935 (0.935/0.068) & 0.712 & 0.915 (0.925/0.055) & 0.690 & 0.990 \\
      Highpass (1500Hz) & \textbf{0.998 (0.998/0.002)} & \textbf{0.983} & 0.997 (0.997/0.002) & 0.981 & 0.875 (0.875/0.095) & 0.612 & 0.855 (0.880/0.065) & 0.595 & 0.991 \\
      Lowpass (500Hz) & \textbf{1.000 (1.000/0.000)} & \textbf{0.982} & \textbf{1.000 (1.000/0.000)} & \textbf{0.982} & 0.978 (0.978/0.038) & 0.882 & 0.970 (0.975/0.032) & 0.865 & 0.993\\
      Speed (1.25$\times$) & \textbf{1.000 (1.000/0.000)} & \textbf{0.983} & 0.999 (0.999/0.001) & 0.980 & 0.957 (0.957/0.087) & 0.903 & 0.940 (0.950/0.060) & 0.890 & 0.992 \\
      Resample (32kHz) & \textbf{1.000 (1.000/0.000)} & \textbf{0.986} & \textbf{1.000 (1.000/0.000)} & \textbf{0.986} & 0.975 (0.975/0.072) & 0.875 & 0.960 (0.970/0.045) & 0.860 & 0.995 \\
      Boost (Factor 10) & \textbf{1.000 (1.000/0.000)} & \textbf{0.986} & \textbf{1.000 (1.000/0.000)} & \textbf{0.986} & \textbf{1.000 (1.000/0.000)} & 0.895 & \textbf{1.000 (1.000/0.000)} & 0.870 & 0.994 \\
      Duck (Factor 0.1) & \textbf{1.000 (1.000/0.000)} & \textbf{0.985} & \textbf{1.000 (1.000/0.000)} & \textbf{0.985} & \textbf{1.000 (1.000/0.000)} & 0.892 & \textbf{1.000 (1.000/0.000)} & 0.868 & 0.994 \\
      Echo (0.5s, 0.5v) & 0.999 (0.999/0.001) & \textbf{0.984} & 0.998 (0.998/0.002) & 0.982 & \textbf{1.000 (1.000/0.000)} & 0.900 & 0.930 (0.890/0.030) & 0.880 & 0.998 \\
      Pink Noise ($\sigma$=0.1) & \textbf{1.000 (1.000/0.000)} & \textbf{0.986} & \textbf{1.000 (1.000/0.000)} & \textbf{0.986} & \textbf{1.000 (1.000/0.000)} & 0.885 & 0.980 (0.810/0.050) & 0.820 & 0.996 \\
      White Noise ($\sigma$=0.05) & \textbf{0.999 (0.999/0.001)} & \textbf{0.982} & \textbf{0.999 (0.999/0.001)} & 0.979 & 0.910 (0.860/0.040) & 0.850 & 0.500 (0.540/0.540) & 0.520 & 0.996 \\
      Smooth (Window 40) & \textbf{1.000 (1.000/0.000)} & \textbf{0.983} & \textbf{1.000 (1.000/0.000)} & \textbf{0.983} & 0.990 (0.990/0.000) & 0.880 & 0.940 (0.930/0.040) & 0.870 & 0.988 \\
      MP3 (32kbps) & 0.999 (0.999/0.001) & \textbf{0.980} & 0.999 (0.999/0.001) & 0.978 & \textbf{1.000 (1.000/0.000)} & 0.875 & \textbf{1.000 (0.990/0.000)} & 0.860 & 0.987 \\
      AAC (64kbps) & \textbf{1.000 (1.000/0.000)} & \textbf{0.981} & \textbf{1.000 (1.000/0.000)} & \textbf{0.981} & \textbf{1.000 (1.000/0.000)} & 0.880 & \textbf{1.000 (1.000/0.000)} & 0.880 & 0.989 \\
      EnCodec (nq=16) & \textbf{0.995 (0.995/0.005)} & \textbf{0.976} & 0.960 (0.960/0.040) & 0.920 & 0.980 (0.980/0.010) & 0.760 & 0.000 (0.000/1.000) & 0.000 & 0.975 \\
      \bottomrule
    \end{tabular}%
  }
  \begin{minipage}{\textwidth}
    \vspace{0.15cm}
    \footnotesize
    \textit{Notes:} Det. = Detection; Extr. Acc. = Extraction Accuracy.  
    AudioAuth achieves highly robust detection (TPR $\geq$ 0.995, FPR $\leq$ 0.005) across all 14 audio effects with MIoU $\geq$ 0.976. WaveVerify shows competitive performance but exhibits slight degradation on aggressive frequency-selective attacks (Bandpass, Highpass), echo effects, and EnCodec compression. AudioSeal exhibits variable robustness, particularly weak to highpass filtering (0.612 MIoU). WavMark completely fails on EnCodec (0.00 TPR/FPR) and white noise (0.50 TPR/FPR). WMCodec achieves~$>$~0.98 extraction accuracy across all audio effects with particularly strong performance on amplitude operations (Boost/Duck, 0.994) and codec compression attacks (Echo 0.998, Noise 0.996).
  \end{minipage}
\end{table*}

Notably, all results in Table~II were obtained on RAVDESS (emotional speech) and ASVspoof 2019 (replay and spoofing attacks), datasets entirely unseen during training, demonstrating cross-domain generalization from the LibriSpeech/CommonVoice/CMU-ARCTIC/DiPCo training distribution to out-of-domain emotional and adversarial speech conditions.
This broad robustness is primarily attributable to the \emph{augmentation training strategy}: the dynamic effect scheduler (Algorithm~1) progressively increases exposure to the most challenging distortions based on real-time BER and MIoU feedback, enabling the generator and detector to co-adapt to adversarial conditions. The complementary band allocation (Section~III-A) provides an additional resilience layer by ensuring that frequency-selective attacks on one band preserve recoverability from the other. A dedicated lightweight Locator network ($0.13$M parameters) identifies watermark boundaries with detection windows as small as 50ms. This robustness profile ensures reliable watermark detection and content-integrity verification across the signal transformations routinely encountered in telephony-grade voice biometric deployments, including codec compression, bandwidth limitation, and ambient noise contamination~\cite{das2020assessing}.
\vspace{-2.5mm}
\subsection{Resilience Against Temporal Attacks, Combined Effects, and Adversarial Attacks}
To assess resilience against temporal reordering attacks, we tested our model on audio reversal, circular shifting, and segment shuffling. Figure~5 shows that AudioAuth maintains near-zero BER (0.009 $\pm$ 0.002) across all temporal attacks, achieving 58× improvement over AudioSeal (average temporal BER 0.52 vs.\ 0.009). Under reversal specifically, AudioSeal degrades to BER 0.56 $\pm$ 0.02, its worst case, due to frame-alignment assumptions collapsing under temporal desynchronization~\cite{audioseal}. WaveVerify achieves moderate resilience (BER: 0.07 $\pm$ 0.011) through FiLM-based conditioning but lags AudioAuth by 8× due to its single-channel design lacking frequency-partitioned redundancy.

AudioAuth's superior performance stems from: (1) multi-scale FiLM-based embedding across hierarchical encoder layers (0--3, downsampling [8,5,4,2]) that preserves coarse embeddings through global temporal modifications while fine embeddings resist local perturbations; and (2) targeted temporal augmentations during training (reversals, rotations, shuffles) that force learning of position-independent watermark features. The dual-channel architecture provides inherent redundancy: attacks corrupting one frequency partition leave the complementary partition intact, ensuring robust recovery under partial disruption.

While WavMark provides segment-level localization~\cite{wavmark}, AudioAuth, WaveVerify, and AudioSeal all achieve sample-wise localization with superior precision in identifying tampered regions. Representation-level baselines (DiscreteWM, Timbre Watermarking, WMCodec) and attribution-specific methods (FakeMark, AudioMarkNet) are excluded from this comparison as they fundamentally lack the architectural capacity for fine-grained sample-wise temporal boundary detection. For voice biometric applications, this temporal resilience is particularly relevant: replay attacks and splicing-based impersonation attempts commonly involve temporal manipulations (segment reordering, concatenation of utterance fragments) that would defeat watermarking methods lacking position-independent feature learning~\cite{asvspoof2019}.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{IEEEtran/figures/watermark_decoding_accuracy.png}
    \caption{Watermark decoding accuracy under temporal attacks. AudioAuth achieves near-zero BER (0.009 $\pm$ 0.002) across reversal, circular shift, and segment shuffle, outperforming AudioSeal (0.52 $\pm$ 0.019) and WaveVerify (0.07 $\pm$ 0.011) by 58× and 8×, respectively. The superiority is attributed to frequency-partitioned FiLM-based hierarchical embedding and targeted temporal augmentations enabling position-independent feature learning. Error bars represent standard deviations over three trials on 1,000 RAVDESS and ASVspoof 2019 audio clips.}
    \label{fig:watermark_decoding_accuracy}
    \vspace{-0.5\baselineskip} 
\end{figure}

To evaluate resilience under realistic conditions, we tested seven \textbf{combined audio effect scenarios} (e.g., codec+noise, filtering+speed change). AudioAuth achieves 0.995 average detection rate and 0.957 MIoU across all combinations, substantially outperforming WaveVerify (0.986/0.945), AudioSeal (0.737/0.521), and WavMark (0.677/0.478). These results demonstrate robust watermark detection and recovery even when audio traverses multiple processing stages common in voice biometric pipelines. Complete per-scenario results are provided in Supplementary Section~II-B (Table~VI).

While audio processing transformations represent natural distortions,
\textbf{adversarial attacks} pose a more sophisticated threat where attackers intentionally craft perturbations to evade or forge watermarks~\cite{chen2019hopskipjump, andriushchenko2020square}. To evaluate AudioAuth's security, we assess robustness against black-box and white-box attacks implemented using the AudioMarkBench framework~\cite{liu2024audiomarkbench}. For black-box scenarios, we employ \textit{HopSkipJump Attack} (HSJA)~\cite{chen2019hopskipjump}, an iterative decision-based boundary attack, and \textit{Square Attack}~\cite{andriushchenko2020square}, a query-efficient randomized search method. For white-box scenarios, we utilize gradient-based optimization~\cite{madry2018towards}, granting attackers full model access to minimize the watermark detection loss. All attacks are constrained to maintain perceptual quality (ViSQOL $\geq$ 3.0), ensuring realistic threat modeling where content usability is preserved.
We specifically focus on adversarial attacks that maintain perceptual audio quality (ViSQOL $\geq$ 3.0), representing realistic scenarios where attackers seek to remove watermarks while preserving content usability.

\begin{table*}[htbp]
\centering
\caption{Robustness Evaluation Against Adversarial Attacks. Black-box attacks use 10,000 iterations. White-box attacks use gradient-based optimization with SNR (Signal-to-Noise Ratio) constraints. Bold values indicate best performance for each attack scenario.}
\label{tab:adversarial-attacks}
\small
\renewcommand{\arraystretch}{1.35}
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{%
    \begin{tabular}{l|ccc|ccc|ccc|ccc}
      \toprule
      \multirow{2}{*}{\textbf{Adversarial Attack}}
      & \multicolumn{3}{c|}{\textbf{AudioAuth (Ours)}}
      & \multicolumn{3}{c|}{\textbf{WaveVerify}}
      & \multicolumn{3}{c|}{\textbf{AudioSeal}}
      & \multicolumn{3}{c}{\textbf{WavMark}} \\
      \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13}
      & \textbf{FNR (\%)} & \textbf{SNR} & \textbf{ViSQOL}
      & \textbf{FNR (\%)} & \textbf{SNR} & \textbf{ViSQOL}
      & \textbf{FNR (\%)} & \textbf{SNR} & \textbf{ViSQOL}
      & \textbf{FNR (\%)} & \textbf{SNR} & \textbf{ViSQOL} \\
      \midrule
      \multicolumn{13}{l}{\textit{Black-box Watermark Removal}} \\
      \quad HSJA - Waveform (10k iter.)
        & \textbf{15.2} & 25.8 & 3.60
        & 22.4 & 26.5 & 3.70
        & 52.7 & 27.3 & 3.70
        & 98.4 & 36.5 & 4.30 \\
      \quad HSJA - Spectrogram (10k iter.)
        & \textbf{8.7} & 26.1 & 3.70
        & 14.8 & 27.2 & 3.80
        & 42.3 & 28.7 & 3.80
        & 95.8 & 38.2 & 4.40 \\
      \quad Square Attack - $\ell_\infty$ (10k iter.)
        & \textbf{12.4} & 23.5 & 3.50
        & 19.6 & 24.1 & 3.60
        & 38.5 & 24.9 & 3.60
        & 100.0 & 32.6 & 4.10 \\
      \midrule
      \multicolumn{13}{l}{\textit{Black-box Watermark Forgery}} \\
      \quad HSJA - Spectrogram (10k iter.)
        & \textbf{0.3} & 21.3 & 3.00
        & 0.5 & 20.8 & 3.00
        & 3.2 & 18.5 & 2.80
        & 0.8 & 22.4 & 3.10 \\
      \quad Square Attack - $\ell_\infty$ (10k iter.)
        & \textbf{0.6} & 18.7 & 2.90
        & 0.9 & 18.2 & 2.90
        & 5.7 & 16.2 & 2.60
        & 1.2 & 19.8 & 3.00 \\
      \midrule
      \multicolumn{13}{l}{\textit{White-box Watermark Removal}} \\
      \quad Gradient-based (SNR=20)
        & \textbf{87.4} & 20.0 & 3.20
        & 94.2 & 20.0 & 3.20
        & 100.0 & 20.0 & 3.30
        & 100.0 & 20.0 & 3.10 \\
      \quad Gradient-based (SNR=30)
        & \textbf{52.3} & 30.0 & 3.90
        & 68.7 & 30.0 & 3.90
        & 98.5 & 30.0 & 3.80
        & 100.0 & 30.0 & 3.90 \\
      \quad Gradient-based (SNR=40)
        & \textbf{18.6} & 40.0 & 4.30
        & 32.5 & 40.0 & 4.30
        & 82.3 & 40.0 & 4.20
        & 95.7 & 40.0 & 4.40 \\
      \quad Gradient-based (SNR=50)
        & \textbf{4.2} & 50.0 & 4.60
        & 12.8 & 50.0 & 4.60
        & 45.2 & 50.0 & 4.50
        & 68.4 & 50.0 & 4.70 \\
      \midrule
      \multicolumn{13}{l}{\textit{White-box Watermark Forgery}} \\
      \quad Gradient-based (SNR=30)
        & \textbf{41.5} & 30.0 & 3.80
        & 58.3 & 30.0 & 3.80
        & 94.7 & 30.0 & 3.90
        & 100.0 & 30.0 & 3.80 \\
      \quad Gradient-based (SNR=40)
        & \textbf{12.3} & 40.0 & 4.20
        & 24.6 & 40.0 & 4.20
        & 76.3 & 40.0 & 4.30
        & 97.2 & 40.0 & 4.50 \\
      \quad Gradient-based (SNR=50)
        & \textbf{2.9} & 50.0 & 4.60
        & 8.4 & 50.0 & 4.60
        & 42.1 & 50.0 & 4.60
        & 82.5 & 50.0 & 4.70 \\
      \bottomrule
    \end{tabular}%
  }
\end{table*}

Table~III demonstrates AudioAuth’s strong resilience across all evaluated threat models. Following the AudioMarkBench framework~\cite{liu2024audiomarkbench}, we evaluate representative black-box attacks (HopSkipJump, Square Attack) and gradient-based white-box optimization as a worst-case baseline. Under spectrogram-domain HSJA, AudioAuth attains an FNR of 8.7\%, compared to 42.3\% for AudioSeal and 95.8\% for WavMark. In white-box settings at practical perturbation budgets (SNR=40, ViSQOL $\approx$ 4.3), AudioAuth achieves 18.6\% FNR versus 82.3\% and 95.7\% for AudioSeal and WavMark, respectively (a 4.4$\times$ improvement). At near-imperceptible perturbations (SNR=50), AudioAuth maintains 4.2\% FNR, while AudioSeal and WavMark degrade to 45.2\% and 68.4\%.
Cross-attack evaluation shows strong generalization: models trained on HSJA exhibit only a 2.4-point FNR increase under the unseen Square Attack, indicating transferable adversarial robustness.

Critically, AudioAuth is the only method maintaining \emph{acceptable robustness} (FNR $<$ 20\%) under white-box attacks at realistic budgets, enabling open-sourcing of detector weights without catastrophic degradation. This advantage is attributed to the frequency-partitioned FiLM architecture’s complementary band weighting. While all methods fail under extreme white-box access (e.g., 87.4\% FNR at SNR=20,dB), such assumptions are unrealistic in deployment. In operational voice biometric settings, the black-box threat model is most relevant, where AudioAuth’s 8.7–15.2\% FNR provides a strong deterrent against provenance stripping~\cite{liu2024audiomarkbench}.

Additionally, evaluation on the ReplayDF dataset~\cite{muller2025replaydf}, comprising 4,000 physically replayed deepfake recordings across 109 loudspeaker--microphone configurations and four unseen TTS systems, demonstrates that \emph{AudioAuth reliably embeds and detects watermarks on replay-degraded synthetic audio}. AudioAuth achieves the highest average MIoU of 0.958 among all evaluated methods, confirming its robustness under realistic physical replay conditions. Detailed per-system results are reported in Supplementary Section~II-E (Table~IX).

\subsection{Multi-Source Attribution Accuracy}

A key application for voice biometrics security is attributing synthetic speech to its generating model, enabling forensic investigation and accountability when voice spoofing or impersonation is suspected. We evaluated AudioAuth's source attribution capability against WaveVerify~\cite{waveverify2025} and AudioSeal~\cite{audioseal}, both of which embed a single general-purpose 16-bit payload repurposed here for model identification, whereas AudioAuth dedicates one of its two 16-bit channels exclusively to model identity while the second channel preserves integrity-verification capability. Each system was evaluated by assigning unique 16-bit model identifiers to eight distinct TTS systems (VALL-E~\cite{valle}, XTTS~\cite{xtts2024}, Bark, Tacotron2~\cite{tacotron2}, FastSpeech2~\cite{fastspeech2}, VITS~\cite{vits2021}, NaturalSpeech2~\cite{naturalspeech2}, Tortoise~\cite{tortoisetts}) and measuring identification accuracy on 1,000 samples per system, drawn from the RAVDESS~\cite{ravdess} and ASVspoof 2019 LA~\cite{asvspoof2019} evaluation sets, under various distortion conditions.

\begin{table}[htbp]
\centering
\caption{Multi-Source Attribution Accuracy (\%). 8 TTS systems with unique 16-bit watermark identifiers. Accuracy measured as correct source identification rate across 1,000 samples per system. Bold values indicate best performance per condition.}
\label{tab:multi-source-attribution}
\small
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Condition} & \textbf{AudioAuth} & \textbf{WaveVerify} & \textbf{AudioSeal} \\
\midrule
Clean (no attack) & \textbf{99.8} & 99.4 & 98.7 \\
MP3 128kbps & \textbf{99.2} & 98.1 & 96.3 \\
AAC 64kbps & \textbf{98.9} & 97.4 & 94.8 \\
Highpass 1500Hz & \textbf{97.6} & 96.2 & 78.4 \\
White noise ($\sigma$=0.02) & \textbf{98.4} & 97.8 & 91.2 \\
Speed 1.1$\times$ & \textbf{99.1} & 98.6 & 93.7 \\
\midrule
\textbf{Average} & \textbf{98.8} & 97.9 & 92.2 \\
\bottomrule
\end{tabular}
\end{table}

Table~IV shows that AudioAuth achieves 98.8\% average attribution accuracy across distortions, \emph{demonstrating reliable source identification} for forensic analysis. The key enabler is \emph{dedicated band separation}: AudioAuth's dual-channel design physically isolates model-identifier and integrity-verification streams into complementary frequency bands, whereas both WaveVerify and AudioSeal multiplex a single general-purpose 16-bit payload without spectral partitioning. This separation ensures that even when integrity-verification bits are corrupted by aggressive frequency-selective attacks (e.g., highpass filtering reducing AudioSeal to 78.4\%), model identifier bits remain recoverable from their dedicated primary bands. WaveVerify's uniform payload allocation still achieves 97.9\% under clean conditions but degrades more sharply under frequency-selective distortions where dedicated band separation proves critical. The resulting attribution capability is essential for voice biometrics security investigations, where identifying the source TTS system enables tracing attack provenance and establishing accountability~\cite{ge2025fakemark,zong2025audiomarknet}.

\vspace{-2.5mm}
\subsection{Open-set Watermark Provenance Verification}\label{sec:open-set}

To evaluate real-world deployment readiness, we assessed AudioAuth under open-set conditions using 12{,}000 samples across three categories: watermarked synthetic speech from eight known TTS systems, genuine human speech, and unwatermarked synthetic speech from four unseen third-party TTS systems. AudioAuth achieves 99.5\% overall accuracy on mixed unauthorized sources under clean conditions, outperforming WaveVerify (98.9\%) and AudioSeal (97.1\%). Under distortions, performance degrades gracefully, maintaining 97.1\% accuracy even under high-pass filtering at 1500\,Hz. These results confirm that AudioAuth performs provenance verification rather than passive detection, reliably distinguishing watermarked content from both genuine speech and unknown-pipeline deepfakes. Full per-category results under clean and distorted conditions are provided in Supplementary Table~VII.

\vspace{-2.5mm}
\subsection{Impact on Speaker Verification Systems}

To evaluate the compatibility of AudioAuth with downstream speaker verification systems, we measured the impact of watermark embedding on speaker embedding quality. Since AudioAuth's dual-channel payload embeds both model-identifier bits and integrity-verification bits into every watermarked signal, it is essential to verify that neither channel's spectral footprint degrades the speaker-discriminative features relied upon by verification systems. While the model-identifier channel primarily serves to attribute synthetic speech to its generating TTS system, the same 16-bit field can encode source-device or pipeline identifiers when watermarking genuine recordings within a content provenance framework~\cite{c2pa2024}. Regardless of payload semantics, the watermark’s acoustic impact on speaker embeddings must remain negligible for AudioAuth to be compatible with operational voice biometric infrastructure and positive, accountable
uses of synthetic speech.

Following the standard VoxCeleb1-O speaker verification protocol~\cite{nagrani2020voxceleb}, which comprises 37{,}611 trial pairs from 40 speakers, we extracted 192-dimensional speaker embeddings using a pretrained ECAPA-TDNN model~\cite{Desplanques_2020} and computed cosine-similarity scores for each trial pair. Equal Error Rate (EER) and minimum Detection Cost Function (minDCF, $C_\mathrm{miss}=C_\mathrm{fa}=1$, $P_\mathrm{target}=0.05$) were then measured on the original (unwatermarked) utterances and on their watermarked counterparts to quantify any degradation introduced by the embedding process.

\begin{table}[htbp]
\centering
\caption{Impact of Watermarking on Speaker Verification Performance. EER (\%) measured using ECAPA-TDNN embeddings on VoxCeleb1-O test set. $\Delta$EER indicates degradation from unwatermarked baseline. Bold values indicate best performance (lowest degradation).}
\label{tab:speaker-verification-impact}
\small
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Method} & \textbf{EER (\%)} & \textbf{$\Delta$EER} & \textbf{minDCF} \\
\midrule
Unwatermarked (Baseline) & 2.31 & --- & 0.142 \\
\midrule
AudioAuth (Ours) & \textbf{2.38} & \textbf{+0.07} & \textbf{0.148} \\
WaveVerify & 2.45 & +0.14 & 0.156 \\
AudioSeal & 2.89 & +0.58 & 0.201 \\
WavMark & 3.12 & +0.81 & 0.224 \\
WMCodec & 4.67 & +2.36 & 0.312 \\
Timbre & 5.84 & +3.53 & 0.398 \\
\bottomrule
\end{tabular}
\end{table}

Table~V shows that AudioAuth introduces minimal speaker verification degradation ($\Delta$EER = +0.07\%), substantially outperforming AudioSeal (+0.58\%) and WMCodec (+2.36\%). This preservation of speaker discriminability is attributed to AudioAuth's \emph{minimal spectral footprint}: the complementary band weighting distributes watermark energy thinly across all four frequency bands rather than concentrating it in any single region, avoiding systematic corruption of speaker-characteristic formants (F1--F4, primarily 300--3500\,Hz) that speaker embedding extractors rely upon. In contrast, methods with heavier per-band energy concentration (e.g., WMCodec's codec-bottleneck embedding) introduce broader spectral distortions that degrade speaker embeddings, demonstrating that AudioAuth can be deployed in tandem with speaker verification systems without compromising speaker discriminability~\cite{tomi2021tandem}.
\vspace{-5mm}
\section{Conclusion}\label{sec:conclusion}
In this paper, we presented AudioAuth, a dual-channel audio watermarking framework designed to jointly support source attribution and content-integrity verification under realistic threat models. AudioAuth departs from conventional single-payload watermarking by embedding model-identification and integrity-verification information through a frequency-partitioned FiLM-based design, enabling complementary recovery under spectral attenuation. A decoupled locator–detector architecture trained with adversarial distortions enables reliable watermark embedding, localization, and detection without reliance on fragile synchronization mechanisms.

Extensive evaluation demonstrates reliable watermark embedding and detectability across diverse signal-level distortions and adversarial conditions, while maintaining high localization accuracy under temporal desynchronization. Experiments on physically replayed deepfakes spanning 109 loudspeaker–microphone configurations show that watermarking remains effective when applied to synthetic audio already degraded by acoustic replay. In addition, multi-source experiments across eight text-to-speech systems confirm accurate source attribution, while open-set evaluation verifies robust discrimination between provenance-verifiable content and audio from unknown or non-embedding pipelines. Collectively, these results establish \emph{AudioAuth as a practical and attack-resilient watermarking framework} for audio provenance and integrity verification in real-world settings.

%AudioAuth addresses the challenge of protecting voice biometric systems against synthetic speech threats by unifying \emph{source attribution} and \emph{speech content integrity verification} within a single watermarking framework. As speaker verification systems face escalating attacks from text-to-speech and voice conversion technologies—well documented through the ASVspoof challenge series—proactive watermarking provides an orthogonal defense that complements passive anti-spoofing countermeasures. The core design—frequency-partitioned FiLM-based embedding with complementary 70/30 band weighting—distributes model-identifier and integrity-verification bits across even and odd frequency bands, ensuring that frequency-selective attacks targeting one channel preserve recoverable information in adjacent bands. Combined with a decoupled locator–detector architecture and dynamic adversarial training, AudioAuth achieves state-of-the-art or competitive performance across six baselines, including MIoU of $0.983$ under high-pass filtering, BER of $0.009\%$ under temporal attacks, and strong adversarial resilience, while maintaining high perceptual quality (STOI $0.96$, ViSQOL $4.76$).

\textbf{Implications for Voice Biometric Security and Media Integrity.}
AudioAuth provides a practical provenance layer for voice biometric systems by enabling persistent watermarking of synthetic speech while leaving bona fide human speech unmodified. Open-set evaluation shows that AudioAuth reliably distinguishes watermarked synthetic audio from both genuine speech and synthetic audio generated by unknown pipelines, achieving over 99\% discrimination accuracy across clean and distorted conditions. In a tandem deployment, AudioAuth can operate as a front-end module to automatic speaker verification (ASV), where inputs are first checked for a valid watermark prior to identity verification.

Speaker verification experiments on the VoxCeleb1-O protocol confirm that AudioAuth introduces negligible degradation to downstream ASV performance, validating compatibility with existing biometric pipelines. Beyond real-time identity authentication, robust watermark detection, content-integrity verification, accurate multi-source attribution across eight text-to-speech systems, and resilience to replay and signal distortions support post-hoc forensic analysis and accountability. More broadly, AudioAuth contributes a signal-level foundation for audio provenance verification aligned with standards such as C2PA~\cite{C2PA2025}, addressing evaluation gaps highlighted in recent systematizations of audio watermarking~\cite{SoK2024}.

%\textbf{Implications for Voice Biometrics and Media Integrity.}
%AudioAuth enables text-to-speech and voice synthesis providers to embed persistent provenance signatures that remain detectable after compression, filtering, and replay, conditions routinely encountered in call center authentication, mobile voice banking, and VoIP identity verification. For non-watermarked audio (e.g., genuine human speech), the detector outputs near-zero confidence scores, achieving 99.7\% discrimination accuracy between watermarked and unwatermarked content. In a tandem deployment~\cite{tomi2021tandem}, the watermark detector serves as a pre-screening module alongside the ASV system: incoming speech is first checked for a valid watermark signature, and only verified content proceeds to speaker verification, while unverified or tampered audio is flagged for further inspection. Because verification relies on embedded provenance rather than synthesis artifacts, this approach remains effective as generative models evolve. Beyond voice biometrics, the same mechanism supports broader digital media integrity by enabling audio provenance verification in alignment with emerging standards such as C2PA~\cite{C2PA2025}. To our knowledge, AudioAuth is the first framework to jointly evaluate source attribution and content integrity verification specifically for voice biometric scenarios, providing a standardized dual-channel evaluation across eight TTS systems with AudioMarkBench-compatible benchmarking~\cite{liu2024audiomarkbench}, addressing a gap identified by recent systematizations of audio watermarking~\cite{SoK2024}.

\textbf{Limitations and Scope.}
Several limitations merit consideration. First, semantic-loop attacks
(ASR$\rightarrow$TTS) remove signal-level watermarks by regenerating audio from its semantic content rather than modifying the original waveform, while preserving linguistic information; addressing such attacks requires complementary text-level or multimodal provenance mechanisms beyond the scope of signal-domain watermarking. Second, extreme physical-channel conditions—such as over-the-air transmission beyond 2\,m in highly reverberant environments (RT60 $>$ 0.8\,s) or playback through severely degraded loudspeaker hardware—remain unexplored and may reduce watermark recoverability below
acceptable thresholds. While our ReplayDF evaluation captures controlled replay scenarios, unconstrained far-field conditions are not yet addressed. Third, voice conversion pipelines that explicitly alter speaker identity may redistribute spectral energy across frequency bands in ways not yet fully characterized; preliminary analysis suggests that energy redistribution exceeding 15\,dB across adjacent bands could disrupt the complementary allocation that
AudioAuth relies upon for recovery. Finally, watermarking is not a replacement for passive anti-spoofing or speaker verification systems; rather, it serves as a complementary, proactive defense by providing verifiable provenance. These limitations define the current
threat model and motivate future research on cross-modal provenance and physical-channel robustness.

\textbf{Broader Impacts and Societal Considerations.}
AudioAuth aims to improve transparency and traceability in AI-generated audio by enabling reliable provenance verification, aligning with emerging regulatory frameworks such as the EU AI Act that emphasize labeling and accountability for synthetic media. By encoding
source-attribution and integrity-verification information---rather than speaker identity---AudioAuth supports provenance verification without exposing biometric attributes or personal identity.
Audio watermarking is a dual-use technology and therefore requires appropriate safeguards
and governance to prevent unintended or overly restrictive applications, such as misuse in surveillance contexts or inflexible enforcement of copyright on user-generated content.
These considerations highlight the need for transparent deployment policies, open and auditable tooling, clear legal and ethical frameworks, and systematic fairness evaluation
across languages, accents, and speaker demographics prior to large-scale deployment, as the current ten-language training corpus has not yet been assessed for disparities across underrepresented groups.

\textbf{Future Directions.}
Future work will extend AudioAuth to more challenging over-the-air re-recording conditions,
including explicit evaluation of watermark survival when watermarked audio is physically
replayed through loudspeaker–microphone channels. Additional directions include analysis
under voice conversion pipelines, cross-lingual settings, and fairness-stratified evaluation
of watermark impact across diverse speaker embedding architectures. 
%We have publicly released trained models, code and evaluation scripts~\cite{SoK2024} to support reproducibility and establish a standardized
%dual-channel watermarking baseline for the community.\footnote{Code and models: \url{https://github.com/vcbsl/AudioAuth}}

\bibliographystyle{IEEEtran}
\bibliography{Transactions-Bibliography/IEEEabrv,Transactions-Bibliography/egbib}

\section{Biography Section}
\vspace{-5mm}
\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{IEEEtran/figures/aditya_photo.jpg}}]{Aditya Pujari}
is a Ph.D. Scholar in the Department of Computer Science at the University of North Texas (UNT), where he works as a Research Assistant in the Visual Computing and Biometric Security Lab (VCBSL). He received his M.S. degree in Artificial Intelligence from UNT. His research interests include audio watermarking, deepfake detection, media authentication, generative AI, and biometric system security.
\end{IEEEbiography}

\vspace{-5mm}
\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{IEEEtran/figures/professor.jpeg}}]{Ajita Rattani}
is an Assistant Professor in the Department of Computer Science and Engineering at the University of North Texas, Denton, USA. She received her Ph.D. from the University of Cagliari, Italy, and completed her postdoctoral research at Michigan State University, USA. Her research interests include computer vision, image analysis, media forensics, and biometric systems.

Dr. Rattani is an Education Subcommittee Member of the IEEE Biometrics Council and serves as Chair of the IAPR Technical Committee on Biometrics (TC4) Education Subcommittee. She is also the IEEE ISATC Task Force Chair on Biometrics. 
%She has received Best Reviewer Awards from the Elsevier Image and Vision Computing journal in 2018 and from IEEE IJCB in 2021. 
Dr. Rattani currently serves as an Associate Editor for IEEE Transactions on Biometrics, Behavior, and Identity Science (T-BIOM), IEEE Open Journal of the Computer Society (OJCS), and Elsevier Engineering Applications of Artificial Intelligence (EAAI).

\end{IEEEbiography}

\end{document}

