% ------------ identification --------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{jku-icg-thesis}
    [2023/06/30 (Andreas Hinterreiter) Class for JKU ICG Theses]

\LoadClass[11pt, twoside]{report}
\RequirePackage{expl3}

% -------------- languages -----------------
\RequirePackage[
    ngerman,
    english
    ]{babel}

% ------------ font settings ---------------

\RequirePackage[letterspace=150]{microtype}
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{libertinus}
\RequirePackage[amsthm]{libertinust1math}
\RequirePackage[
    scale=0.9
    ]{montserrat}
\RequirePackage[medium,scale=0.81]{FiraSans}
\RequirePackage[scale=0.95]{inconsolata}
%\RequirePackage[sf, bf, medium]{titlesec}

\makeatletter

%%% Font sizes %%%

\renewcommand\normalsize{%
   \@setfontsize\normalsize\@xipt{14.5}%
   \abovedisplayskip 11\p@ \@plus3\p@ \@minus6\p@
   \abovedisplayshortskip \z@ \@plus3\p@
   \belowdisplayshortskip 6.5\p@ \@plus3.5\p@ \@minus3\p@
   \belowdisplayskip \abovedisplayskip
   \let\@listi\@listI}
\normalbaselineskip=14.5pt
\normalsize
\renewcommand\small{%
   \@setfontsize\small\@xpt{12}
   \abovedisplayskip 10\p@ \@plus2\p@ \@minus5\p@
   \abovedisplayshortskip \z@ \@plus3\p@
   \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 6\p@ \@plus2\p@ \@minus2\p@
               \parsep 3\p@ \@plus2\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}
\renewcommand\footnotesize{%
   \@setfontsize\footnotesize\@ixpt{11}%
   \abovedisplayskip 8\p@ \@plus2\p@ \@minus4\p@
   \abovedisplayshortskip \z@ \@plus\p@
   \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 4\p@ \@plus2\p@ \@minus2\p@
               \parsep 2\p@ \@plus\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}
\renewcommand\scriptsize{\@setfontsize\scriptsize\@viiipt\@xpt}
\renewcommand\tiny{\@setfontsize\tiny\@vipt\@viipt}
\renewcommand\large{\@setfontsize\large\@xiipt{16}}
\renewcommand\Large{\@setfontsize\Large\@xivpt{18}}
\renewcommand\LARGE{\@setfontsize\LARGE\@xviipt{23}}
\renewcommand\huge{\@setfontsize\huge\@xxpt{30}}
\renewcommand\Huge{\@setfontsize\Huge{25}{36}}

\makeatother

% -------------- hyperref ------------------

\makeatletter

\AtEndPreamble{
    \usepackage{hyperref}
    \hypersetup{
        colorlinks=true,
        allcolors=[rgb]{0.05,0.41,0.53},
    }

    \urlstyle{tt}

    \g@addto@macro\UrlSpecials{\camelurl}
    \def\camelurl{%
    \count@`a
    \loop
    \mathcode\count@"8000
    \uccode`\~\count@\uppercase{\edef~{\mathchar\the\count@\noexpand\breakifupper}}%
    \ifnum\count@<`\z
    \advance\count@\@ne
    \repeat}

    \def\breakifupper#1{%
    \ifcat .\noexpand#1%
    \ifnum`#1>40
    \ifnum`#1<91
    \penalty\z@
    \fi\fi\fi
    #1%
    }
}

\makeatother
    
% ------------ page geometry ---------------

\RequirePackage{geometry}
\geometry{
    a4paper,
    left=9pc,
    right=9pc,
    top=6pc,
    headsep=2\baselineskip,
    textheight=47\baselineskip,
    headheight=\baselineskip,
}

% --------------- floats -------------------

\RequirePackage{caption}
\captionsetup[figure]{
    labelfont={bf,sf},
    textfont={sf},
    margin=1.0em,
    justification=RaggedRight,
    name={Fig.},
    skip=2.5ex,
}
\captionsetup[table]{
    labelfont={bf,sf},
    textfont={sf},
    margin=1.0em,
    justification=RaggedRight,
    skip=2.5ex,
}
\makeatletter
\let\tableorig\table
\def\table@i[#1]{\tableorig[#1]\sffamily}  % with optional argument
\def\table@ii{\tableorig\sffamily}  % without optional argument
\def\table{\@ifnextchar[\table@i \table@ii}  % Redefine depending on presence of [
\makeatother

%%% Table of contents %%%

\RequirePackage[titles]{tocloft}
\setcounter{tocdepth}{1}

\renewcommand{\cftchapfont}{\raggedright\normalfont}
\renewcommand{\cftchapleader}{\quad}
\renewcommand{\cftchapafterpnum}{\cftparfillskip}
\renewcommand{\cftpnumalign}{l}
\renewcommand{\cftchappagefont}{\normalfont}

\renewcommand{\cftsecfont}{\raggedright\normalfont}
\renewcommand{\cftsecleader}{\quad}
\renewcommand{\cftsecafterpnum}{\cftparfillskip}
\renewcommand{\cftsecpagefont}{\normalfont}

%%% Chapter and section headings %%%

\RequirePackage[explicit]{titlesec}

\newcommand{\jkuicgchapterfont}{\Huge\firasemibold}
\newcommand{\jkuicgsectionfont}{\LARGE\firasemibold}
\newcommand{\jkuicgsubsectionfont}{\large\firamedium}
\newcommand{\jkuicgparagraphfont}{\large\firamedium}

\titleformat{\chapter}%
	[hang]% shape
	{\raggedright}% format
	{\jkuicgchapterfont\thechapter}% chapter number
	{1em}% spacing
	{% before title
	\jkuicgchapterfont #1}%
	[]

\titleformat{name=\chapter, numberless}%
	[hang]% shape
	{\raggedright}% format
	{}% chapter number
	{0pt}% spacing
	{% before title
	\jkuicgchapterfont #1}%
	[]

\titleformat{\section}%
	[hang]% shape
	{\raggedright}% format
	{\jkuicgsectionfont \thesection}% chapter number
	{1em}% spacing
	{% before title
	\jkuicgsectionfont #1}%
	[]

\titleformat{name=\section, numberless}%
	[hang]% shape
	{\raggedright}% format
	{}% chapter number
	{0pt}% spacing
	{% before title
	\jkuicgsectionfont #1}%
	[]

 \titleformat{\subsection}%
	[hang]% shape
	{\raggedright}% format
	{\jkuicgsubsectionfont \thesubsection}% chapter number
	{1em}% spacing
	{% before title
	\jkuicgsubsectionfont #1}%
	[]

\titleformat{name=\subsection, numberless}%
	[hang]% shape
	{\raggedright}% format
	{}% chapter number
	{0pt}% spacing
	{% before title
	\jkuicgsubsectionfont #1}%
	[]

\titleformat{name=\paragraph, numberless}
    [runin]% shape
    {}% format
    {}% chapter number
    {0pt}% spacing
    {% before title
    \jkuicgparagraphfont #1}%
    []

\titlespacing*{\chapter}{0pt}{3.5ex plus 1ex minus .2ex}{2.5ex plus .2ex}
\titlespacing*{\section}{0pt}{2.5ex plus 1ex minus .2ex}{2.5ex plus .2ex}
\titlespacing*{\section}{0pt}{2.5ex plus 1ex minus .2ex}{2.5ex plus .2ex}


% ---------- title & abstract --------------

\ExplSyntaxOn

% supervisor handling

\seq_gclear_new:N \jkuicg_supervisors_seq
\seq_gclear_new:N \jkuicg_supervisorroles_seq

\NewDocumentCommand \supervisor { O{Betreuer} m } 
    {
        \seq_put_right:Nn \jkuicg_supervisors_seq {#2}
        \seq_put_right:Nn \jkuicg_supervisorroles_seq {#1}
    }

% thesis properties

\prop_clear_new:N \jkuicg_thesis_prop

\keys_define:nn { jkuicg }
    {
        subject .choice:,
        subject / ai .code:n = {
            \prop_gput:Nnn \jkuicg_thesis_prop { subject } { Artificial\c_space_tl Intelligence }
        },
        subject / cs .code:n = {
            \prop_gput:Nnn \jkuicg_thesis_prop { subject } { Informatik }
            \bool_gset_true:N \jkuicg_der_bool
        },
        subject / tech .code:n = {
            \prop_gput:Nnn \jkuicg_thesis_prop { subject } { Technischen\c_space_tl Wissenschaften}
            \bool_gset_true:N \jkuicg_der_bool
        },
        study .choices:nn = { bsc, msc, phd } { \prop_gput:Nnn \jkuicg_thesis_prop { study } { #1 } },
        institute .code:n = \prop_gput:Nnn \jkuicg_thesis_prop { institute } { #1 },
        location .code:n = \prop_gput:Nnn \jkuicg_thesis_prop { location } { #1 },
        month .code:n = \prop_gput:Nnn \jkuicg_thesis_prop { month } { #1 },
        year .code:n = \prop_gput:Nnn \jkuicg_thesis_prop { year } { #1 },
        twoside .bool_gset:N = \jkuicg_twoside_bool,
    }
\keys_set:nn { jkuicg }
    {
        subject = ai,
        month = \c_sys_month_int,
        year = \c_sys_year_int,
        twoside = false,
        institute = Institut\c_space_tl für\c_space_tl Computergrafik,
    }

\NewDocumentCommand \ClearPage { } {
    \bool_if:NTF \jkuicg_twoside_bool
        { \clearpage{\thispagestyle{empty}\cleardoublepage} }
        { \clearpage }
}

\cs_new_protected:Nn \jkuicg_month: {
    \int_case:nnF { \prop_item:Nn \jkuicg_thesis_prop { month } }
    {
        {  1 } { Januar }
        {  2 } { Februar }
        {  3 } { März }
        {  4 } { April }
        {  5 } { Mai }
        {  6 } { Juni }
        {  7 } { Juli }
        {  8 } { August }
        {  9 } { September }
        { 10 } { Oktober }
        { 11 } { November }
        { 12 } { Dezember }
    }
    {Unknown\c_space_tl Month\ClassError {jku-icg-thesis} {Invalid\c_space_tl month\c_space_tl specification} {}}
}

\bool_new:N \jkuicg_der_bool
\bool_set_false:N \jkuicg_der_bool

\cs_new_protected:Nn \jkuicg_der: {
    \bool_if:NTF \jkuicg_der_bool
        { \hbox{}\c_space_tl der}
        { }
}

\cs_new_protected:Nn \jkuicg_thesistype: {
    \str_case_e:nnF { \prop_item:Nn \jkuicg_thesis_prop { study } }
        {
            { bsc } { Bachelorarbeit }
            { msc } { Masterarbeit }
            { phd } { Dissertation }
        }
        {Unknown\c_space_tl Study\ClassError {jku-icg-thesis} {Invalid\c_space_tl study\c_space_tl specification} {}}
}

\cs_new_protected:Nn \jkuicg_degree: {
    \str_case_e:nnF { \prop_item:Nn \jkuicg_thesis_prop { study } }
        {
            { bsc } { Bachelor\c_space_tl of\c_space_tl Science }
            { msc } { Master\c_space_tl of\c_space_tl Science }
            { phd } { Doktor\c_space_tl of\c_space_tl Science }
        }
        {Unknown\c_space_tl Study\ClassError {jku-icg-thesis} {Invalid\c_space_tl study\c_space_tl specification} {}}
}

\cs_new_protected:Nn \jkuicg_study: {
    \str_case_e:nnF { \prop_item:Nn \jkuicg_thesis_prop { study } }
        {
            { bsc } { Bachelorstudium\jkuicg_der: }
            { msc } { Masterstudium\jkuicg_der: }
            { phd } { Doktoratsstudium\jkuicg_der: }
        }
        {Unknown\c_space_tl Study\ClassError {jku-icg-thesis} {Invalid\c_space_tl study\c_space_tl specification} {}}
}

\NewDocumentCommand \thesissetup { m }
    {\keys_set:nn { jkuicg } { #1 }}

\cs_new_protected:Nn \jkuicg_subtitle: { }
\bool_new:N \jkuicg_hassubtitle_bool
\bool_set_false:N \jkuicg_hassubtitle_bool

\NewDocumentCommand \subtitle { m } {
    \cs_gset:Nn \jkuicg_subtitle: {#1}
    \bool_gset_true:N \jkuicg_hassubtitle_bool
}

% maketitle

\makeatletter
\RenewDocumentCommand \maketitle { } {
    \newgeometry{top=1.5cm, right=1.5cm, left=2.5cm, bottom=1.5cm}
    \thispagestyle{empty}
    \begingroup
    \setlength{\parindent}{0pt}
    \fontfamily{Montserrat-TLF}\selectfont
    
    \begin{flushright}
    	\includegraphics[width=5.2cm]{figures/logos/jku-logo}\par
    	\bigskip
    	\bigskip
    	
    	\begin{tabular}{r@{}}
    		Eingereicht\c_space_tl von \\
    		\bfseries \@author \\[1.5ex]
    		Angefertigt\c_space_tl am \\
    		\bfseries \prop_item:Nn \jkuicg_thesis_prop { institute } \\[1.5ex]
    		\seq_item:Nn \jkuicg_supervisorroles_seq {1} \\
    		\bfseries \seq_item:Nn \jkuicg_supervisors_seq {1} \\[1.5ex]
        \end{tabular}
        \par
        \if_int_compare:w \seq_count:N \jkuicg_supervisors_seq > 1
           {    
                \begin{tabular}{r@{}}
                    \seq_item:Nn \jkuicg_supervisorroles_seq {2} \\
                    \bfseries \seq_item:Nn \jkuicg_supervisors_seq {2} \\[2ex]
                \end{tabular}
                \par
            }
        \else:
            \vskip 0.5ex
        \fi:
        \jkuicg_month: \c_space_tl
        \int_eval:n { \prop_item:Nn \jkuicg_thesis_prop { year } }
    \end{flushright}
    \vspace*{3.00cm}
    
    {\raggedright\fontsize{24}{30}\selectfont\bfseries\@title\par}

    \bool_if:NTF \jkuicg_hassubtitle_bool
        {
            \bigskip
            {\Large\bfseries\jkuicg_subtitle:\par}
            \bigskip
        }
        {
            \bigskip
        }
    
    \includegraphics[width=4.4cm]{figures/logos/jku-k}\par
    \bigskip
    \medskip
    
    \begin{tabular}{@{}l}
    	\large \jkuicg_thesistype: \\
    	zur\c_space_tl Erlangung\c_space_tl des\c_space_tl akademischen\c_space_tl Grads \\[.5ex]
    	\large \jkuicg_degree: \\
    	im\c_space_tl\jkuicg_study: \\[.5ex]
    	\large \prop_item:Nn \jkuicg_thesis_prop { subject }
    \end{tabular}
    
    \vspace*{\fill}
    
    \begin{flushright}
    	\begin{tabular}{r@{}}
    		\bfseries JOHANNES\c_space_tl KEPLER \\
    		\bfseries UNIVERSITÄT\c_space_tl LINZ \\
    		Altenberger\c_space_tl Straße\c_space_tl 69 \\
    		4040~Linz,\c_space_tl Österreich \\
    		\href{http://www.jku.at}{www.jku.at}
    	\end{tabular}
    \end{flushright}
    
    \makeatother
    \endgroup
    \restoregeometry
    \ClearPage
}

% affidavit

\makeatletter
\NewDocumentCommand \signaffidavit { O{} } {
    \noindent\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll@{}}
        %\tl_if_empty:nTF #1 { & #1 \\[.5ex] } { }
    	\prop_item:Nn \jkuicg_thesis_prop { location },\c_space_tl
        \jkuicg_month: \c_space_tl
        \int_eval:n { \prop_item:Nn \jkuicg_thesis_prop { year } } &
            \@author
    \end{tabular*}
}
\makeatother

\ExplSyntaxOff

%%% Appendix

\RequirePackage{appendix}
\RequirePackage{bookmark}

\makeatletter
\newenvironment{icgAppendix}{%
    \renewcommand\appendixtocname{Appendix}
    \begin{appendices}
      \bookmarksetupnext{level=-1}
      \addappheadtotoc
      \makeatletter
      \addtocontents{toc}{\let\protect\l@chapter\protect\l@section}
      \addtocontents{toc}{\protect\renewcommand*\protect\l@section[2]{}}
      \makeatother
      \let\oldchap\thechapter
      \renewcommand{\thechapter}{\textsc{\MakeLowercase{\oldchap}}}
}{%
    \end{appendices}
}
\makeatother

