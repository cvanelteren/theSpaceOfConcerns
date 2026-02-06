
 % -------------------------------------------------------------------
%  @LaTeX-class-file{
%     filename        = "fun_article.cls",
%     version         = "1.0",
%     date            = "29 June 2024",
%     codetable       = "ISO/ASCII",
%     keywords        = "LaTeX, fun_article",
%     supported       = "Send email to c.vanelteren@uva.nl",
%     docstring       = "Class for a writing fun aricles."
% --------------------------------------------------------------------
%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{fun_article}[A fun article class]
\LoadClass[a4paper, 9pt, twocolumn]{article}
\usepackage[utf8]{inputenc}
% \usepackage[standard]{ntheorem}
% \usepackage{amsthm}   % Or load amsthm for compatibility
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{wrapfig}
\usepackage{rotating}
\usepackage[normalem]{ulem}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{savetrees}
\usepackage{capt-of}
\usepackage{url}
\def\UrlBreaks{\do\/\do-}
\usepackage[breaklinks]{hyperref}
\newcommand{\fullcitenourl}[2][]{%
\AtNextCitekey{%
\clearfield{url}%
\clearfield{urlyear}%
\clearfield{date}%
\clearfield{urlday}%
\clearfield{urlmonth}%
\clearfield{note}%
\clearfield{eprint}%
\clearfield{doi}%
\clearfield{isbn}%
\clearfield{issn}%
\clearfield{publisher}%
\clearfield{location}%
\clearfield{series}%
\clearfield{volume}%
\clearfield{number}%
\clearfield{pages}%
}%
\ifblank{#1}{%
\citeauthor{#2}. \citetitle{#2} in \citeyear{#2}.%
}{%
\citeauthor{#2}. \citetitle{#2} in  \citeyear{#2}, p. #1.%
}%
}


\usepackage{xr}

\RequirePackage[normalem]{ulem} %DIF PREAMBLE
\RequirePackage{color}\definecolor{RED}{rgb}{1,0,0}\definecolor{BLUE}{rgb}{0,0,1} %DIF PREAMBLE
\providecommand{\DIFadd}[1]{{\protect\color{blue}\uwave{#1}}} %DIF PREAMBLE
\providecommand{\DIFdel}[1]{{\protect\color{red}\sout{#1}}}                      %DIF PREAMBLE
%DIF SAFE PREAMBLE %DIF PREAMBLE
\providecommand{\DIFaddbegin}{} %DIF PREAMBLE
\providecommand{\DIFaddend}{} %DIF PREAMBLE
\providecommand{\DIFdelbegin}{} %DIF PREAMBLE
\providecommand{\DIFdelend}{} %DIF PREAMBLE
%DIF FLOATSAFE PREAMBLE %DIF PREAMBLE
\providecommand{\DIFaddFL}[1]{\DIFadd{#1}} %DIF PREAMBLE
\providecommand{\DIFdelFL}[1]{\DIFdel{#1}} %DIF PREAMBLE
\providecommand{\DIFaddbeginFL}{} %DIF PREAMBLE
\providecommand{\DIFaddendFL}{} %DIF PREAMBLE
\providecommand{\DIFdelbeginFL}{} %DIF PREAMBLE
\providecommand{\DIFdelendFL}{} %DIF PREAMBLE
\providecommand{\DIFDELCMD}{}
%DIF END PREAMBLE EXTENSION ADDED BY LATEXDIFF
% \usepackage[english]{babel} % English language hyphenation

\usepackage{microtype} % Better typography

\usepackage{amsmath,amsfonts,amsthm} % Math packages for equations
% \usepackage[finalizecache,cachedir=.]{minted}
% \usepackage[frozencache,cachedir=.]{minted}
% \usepackage{minted}



\PassOptionsToPackage{svgnames}{xcolor}
\usepackage{awesomebox}
\usepackage{cleveref}
% \usepackage[svgnames]{xcolor} % Enabling colors by their 'svgnames'

\usepackage[size=small, labelfont=bf, up, textfont=it]{caption} % Custom captions under/above tables and figures

\usepackage{booktabs} % Horizontal rules in tables

\usepackage{lastpage} % Used to determine the number of pages in the document (for "Page X of Total")

\usepackage{graphicx} % Required for adding images

\usepackage{enumitem} % Required for customising lists
\setlist{noitemsep} % Remove spacing between bullet/numbered list elements

\usepackage{sectsty} % Enables custom section titles
\allsectionsfont{\usefont{OT1}{phv}{b}{n}} % Change the font of all section commands (Helvetica)

%----------------------------------------------------------------------------------------
%	MARGINS AND SPACING
%----------------------------------------------------------------------------------------


\usepackage{geometry} % Required for adjusting page dimensions

\geometry{
	top=1cm, % Top margin
	bottom=1.5cm, % Bottom margin
	left=2cm, % Left margin
	right=2cm, % Right margin
	includehead, % Include space for a header
	includefoot, % Include space for a footer
	%showframe, % Uncomment to show how the type block is set on the page
}

\setlength{\columnsep}{7mm} % Column separation width

%----------------------------------------------------------------------------------------
%	FONTS
%----------------------------------------------------------------------------------------

\usepackage[T1]{fontenc} % Output font encoding for international characters
\usepackage[utf8]{inputenc} % Required for inputting international characters

\usepackage{XCharter} % Use the XCharter font

%----------------------------------------------------------------------------------------
%	HEADERS AND FOOTERS
%----------------------------------------------------------------------------------------

\usepackage{fancyhdr} % Needed to define custom headers/footers
\pagestyle{fancy} % Enables the custom headers/footers

\renewcommand{\headrulewidth}{0.0pt} % No header rule
\renewcommand{\footrulewidth}{0.4pt} % Thin footer rule

\renewcommand{\sectionmark}[1]{\markboth{#1}{}} % Removes the section number from the header when \leftmark is used

%\nouppercase\leftmark % Add this to one of the lines below if you want a section title in the header/footer

% Headers
\lhead{} % Left header
\chead{\textit{\thetitle}} % Center header - currently printing the article title
\rhead{} % Right header

% Footers
\lfoot{} % Left footer
\cfoot{} % Center footer
\rfoot{\footnotesize Page \thepage\ of \pageref{LastPage}} % Right footer, "Page 1 of 2"

\fancypagestyle{firstpage}{ % Page style for the first page with the title
	\fancyhf{}
	\renewcommand{\footrulewidth}{0pt} % Suppress footer rule
}

%----------------------------------------------------------------------------------------
%	TITLE SECTION
%----------------------------------------------------------------------------------------

\newcommand{\authorstyle}[1]{{\large\usefont{OT1}{phv}{b}{n}\color{DarkRed}#1}} % Authors style (Helvetica)

\newcommand{\institution}[1]{{\footnotesize\usefont{OT1}{phv}{m}{sl}\color{Black}#1}} % Institutions style (Helvetica)

\usepackage{titling} % Allows custom title configuration

\newcommand{\HorRule}{\color{DarkGoldenrod}\rule{\linewidth}{1pt}} % Defines the gold horizontal rule around the title

\pretitle{
	\vspace{-50pt} % Move the entire title section up
	% \HorRule
  % \vspace{5pt} % Horizontal rule before the title
	%\fontsize{32}{36}\usefont{OT1}{phv}{b}{n}\selectfont % Helvetica
  \fontsize{14}{16}\usefont{OT1}{phv}{b}{n}\selectfont % Helvetica
	\color{DarkRed}\raggedright  % Text colour for the title and author(s)
}

\posttitle{
  % \vspace{-10pt}
  \par\vskip 8pt
  % \par\vskip 3pt
} % Whitespace under the title

\preauthor{} % Anything that will appear before \author is printed

\postauthor{ % Anything that will appear after \author is printed
  \par\HorRule
  \vspace{25pt}
}


%----------------------------------------------------------------------------------------
% MATH
%----------------------------------------------------------------------------------------
% \usepackage{amsmath}

%----------------------------------------------------------------------------------------
%	ABSTRACT
%----------------------------------------------------------------------------------------

\usepackage{lettrine} % Package to accentuate the first letter of the text (lettrine)
\usepackage{fix-cm}	% Fixes the height of the lettrine

\newcommand{\initial}[1]{ % Defines the command and style for the lettrine
	\lettrine[lines=3,findent=4pt,nindent=0pt]{% Lettrine takes up 3 lines, the text to the right of it is indented 4pt and further indenting of lines 2+ is stopped
		\color{DarkGoldenrod}% Lettrine colour
		{#1}% The letter
	}{}%
}
\usepackage{xstring} % Required for string manipulation

\newcommand{\lettrineabstract}[1]{
	\StrLeft{#1}{1}[\firstletter] % Capture the first letter of the abstract for the lettrine
	\initial{\firstletter}\textbf{\StrGobbleLeft{#1}{1}} % Print the abstract with the first letter as a lettrine and the rest in bold
}

\makeatletter
\newcommand{\PaperTitle}[1]{\def\@PaperTitle{#1}}
\newcommand{\Archive}[1]{\def\@Archive{#1}}
\newcommand{\Authors}[1]{\def\@Authors{#1}}
\newcommand{\JournalInfo}[1]{\def\@JournalInfo{#1}}
\newcommand{\Abstract}[1]{\def\@Abstract{#1}}
\newcommand{\Keywords}[1]{\def\@Keywords{#1}}
\makeatother


\usepackage{titling,kantlipsum}



\makeatletter
\RequirePackage{environ}
\RenewEnviron{abstract}{%
    \twocolumn[%
        \maketitle
        \begin{@twocolumnfalse}%
            \small
            \vspace{-9em}%
            \begin{list}{}{%
                \setlength{\leftmargin}{5mm}%
                \setlength{\rightmargin}{\leftmargin}%
                \setlength{\topmargin}{0mm}%
            }%
            \item\relax
            \lettrineabstract{\BODY}
            \end{list}%
        \vspace{1em}
        \end{@twocolumnfalse}%
    ]%
}
\newcommand{\makeabstract}{%
    \begin{abstract}
        \@abstractcontent
    \end{abstract}
}

% ---------------------------------------------------------------------


%----------------------------------------------------------------------------------------
%	BIBLIOGRAPHY
%----------------------------------------------------------------------------------------

\usepackage[backend=biber,style=authoryear,natbib=true, style=numeric]{biblatex} % Use the bibtex backend with the authoryear citation style (which resembles APA)
\usepackage[autostyle=true]{csquotes} % Required to generate language-dependent quotes in the bibliography

% \input{fun_article.tex}
\DeclareSourcemap{
  \maps{
    \map{
      \step[fieldsource=url,
            match=\regexp{/books\.google\.},
            fieldset=url, null]
    }
  }
}

\usepackage[english]{babel}
\usepackage{authblk}

\usepackage{cuted}
\usepackage{changepage}
\usepackage{cleveref}
\usepackage{breqn}
\usepackage{chngcntr}
\usepackage{xr}
\usepackage{appendix}

\usepackage{svg}

\usepackage{tikzpagenodes}
\definecolorseries{chcolor}{rgb}{grad}[rgb]{.95,.85,.55}{3,11,17}
\resetcolorseries{chcolor}




\newcommand{\mycitation}[1]{%
This chapter is based on the publication \printbibentry{#1}{author} (\printbibentry{#1}{journal}, \citeyear{#1})%
}

\usepackage{cleveref}
\usepackage{multicol}

% Define a new conditional switch
\newif\ifskippage
% Initialize the switch to false (page is not skipped)
\skippagefalse

\Urlmuskip=0mu plus 1mu\relax

\usepackage[english]{babel}
\usepackage{standalone}
\usepackage{import}
\usepackage{xr}
\usepackage{placeins}
\usepackage{afterpage}
\usepackage{lipsum}
\usepackage{amsmath}
\usepackage{siunitx}

\usepackage{bbm}

\date{}

%% Caption
\RequirePackage{caption}

\DeclareCaptionFormat{firstbold}{%
    #1#2\@makefirstsentencebold{#3}%
}

\newcommand{\@makefirstsentencebold}[1]{%
    \expandafter\@splitatperiod#1. \@endmarker
}

\def\@splitatperiod#1. #2\@endmarker{%
    \def\@testrest{#2}%
    \ifx\@testrest\empty
        % No period-space found, output as-is
        #1%
    \else
        % Period-space found, make first sentence bold
        \textbf{#1.} #2%
    \fi
}

\captionsetup[figure]{
    format=firstbold,
    labelfont=bf,
    textfont=normalfont
}


\setlength{\footskip}{24pt} % increase space between text and footer

\usepackage{sidecap}
\sidecaptionvpos{figure}{c}
