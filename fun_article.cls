
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
\LoadClass[a4paper, 9pt]{article}
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

% ---- Packages moved from document preamble ----
\usepackage{xcolor}
\usepackage{flafter}
\usepackage[most]{tcolorbox}
\usetikzlibrary{arrows.meta,calc,tikzmark}

% ---- Project colour palette ----
\definecolor{DarkRed}{HTML}{8B0000}
\definecolor{DarkGoldenrod}{HTML}{B8860B}
\definecolor{Linen}{HTML}{FAF0E6}
\definecolor{effortcolor}{HTML}{B57F00}
\definecolor{congruencecolor}{HTML}{1B8A3B}
\definecolor{vetocostcolor}{HTML}{B23A2B}

% ---- Significance box ----
\newtcolorbox{significancebox}{
    enhanced,
    colback=Linen!75!white,
    colframe=DarkGoldenrod,
    coltitle=DarkRed,
    title=Significance,
    fonttitle=\bfseries\usefont{OT1}{phv}{b}{n},
    fontupper=\footnotesize,
    boxrule=0.9pt,
    arc=2.5mm,
    left=1.6mm,
    right=1.6mm,
    top=1.0mm,
    bottom=1.0mm,
    attach boxed title to top left={xshift=2.5mm, yshift=-2mm},
    boxed title style={
        colback=black!8,
        colframe=DarkGoldenrod,
        boxrule=0.8pt,
        arc=1.8mm,
        left=1.2mm,
        right=1.2mm,
        top=0.4mm,
        bottom=0.4mm
    }
}

% ---- TikZ equation annotation helpers ----
\newcommand{\eqnmarkbox}[3][black]{%
    \tikzmark{#2-start}\textcolor{#1}{#3}\tikzmark{#2-end}%
}
\newcommand{\annotate}[4][]{%
    \begin{tikzpicture}[remember picture, overlay, >=Stealth]
        \coordinate (#3-mid) at ($(pic cs:#3-start)!0.5!(pic cs:#3-end)$);
        \IfStrEq{#2}{right, above}{%
            \node[anchor=east, font=\small, align=right, #1] (ann-#3) at ($(#3-mid)+(1.6em,1.35em)$) {#4};
            \draw[->, thick] (ann-#3.east) to[out=0, in=60] ($(#3-mid)+(0,1.1ex)$);
        }{}%
        \IfStrEq{#2}{left, above}{%
            \node[anchor=east, font=\small, align=right, #1] (ann-#3) at ($(#3-mid)+(-0.9em,1.35em)$) {#4};
            \draw[->, thick] (ann-#3.east) to[out=0, in=120] ($(#3-mid)+(0,1.1ex)$);
        }{}%
        \IfStrEq{#2}{center, above}{%
            \node[anchor=south, font=\small, align=center, #1] (ann-#3) at ($(#3-mid)+(0,1.35em)$) {#4};
            \draw[->, thick] (ann-#3.south) to ($(#3-mid)+(0,1.1ex)$);
        }{}%
        \IfStrEq{#2}{left, below}{%
            \node[anchor=east, font=\small, align=right, #1] (ann-#3) at ($(#3-mid)+(-0.9em,-1.25em)$) {#4};
            \draw[->, thick] (ann-#3.east) to[out=0, in=240] ($(#3-mid)+(0,-0.8ex)$);
        }{}%
        \IfStrEq{#2}{right, below}{%
            \node[anchor=west, font=\small, align=left, #1] (ann-#3) at ($(#3-mid)+(0.9em,-1.25em)$) {#4};
            \draw[->, thick] (ann-#3.west) to[out=180, in=300] ($(#3-mid)+(0,-0.8ex)$);
        }{}%
    \end{tikzpicture}%
}

% ---- Float tuning ----
\setcounter{topnumber}{1}
\setcounter{dbltopnumber}{1}
\renewcommand{\topfraction}{0.95}
\renewcommand{\dbltopfraction}{0.95}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.85}
\renewcommand{\dblfloatpagefraction}{0.85}
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
	\renewcommand{\footrulewidth}{0.4pt}
	\rfoot{\footnotesize Page \thepage\ of \pageref{LastPage}}
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
    \maketitle
    \thispagestyle{firstpage}%
    \small\relax\vspace{-7em}%
    \begin{list}{}{%
        \setlength{\leftmargin}{5mm}%
        \setlength{\rightmargin}{\leftmargin}%
        \setlength{\topmargin}{0mm}%
    }%
    \item\relax
    \lettrineabstract{\BODY}
    \end{list}%
    \vspace{0.75em}%
}
\newcommand{\makeabstract}{%
    \begin{abstract}
        \@abstractcontent
    \end{abstract}
}

% Optional fixed first-page grid. The first section occupies the left
% column and the sidebar occupies the right; neither column wraps beneath
% the other. \beginarticlecolumns closes the grid and starts page 2.
\newif\if@funsidebar
\@funsidebarfalse
\newif\if@funfirstcolumnsopen
\@funfirstcolumnsopenfalse
\newcommand{\@funsidebarcontent}{}
\newcommand{\sidebar}[1]{%
    \gdef\@funsidebarcontent{#1}%
    \global\@funsidebartrue
}
\newlength{\@funleftcol}
\newlength{\@funrightcol}
\setlength{\@funleftcol}{0.64\textwidth}
\setlength{\@funrightcol}{0.31\textwidth}
\newcommand{\beginarticlecolumns}{%
    \if@funfirstcolumnsopen
        \par
        \end{minipage}%
        \hfill
        \begin{minipage}[t]{\@funrightcol}%
            \vspace{0pt}%
            \@funsidebarcontent
        \end{minipage}%
        \global\@funfirstcolumnsopenfalse
    \fi
    \par\clearpage\twocolumn
}
\newcommand{\@funstartfirstpagecolumns}{%
    \if@funsidebar
        \par\noindent
        \begin{minipage}[t]{\@funleftcol}%
            \vspace{0pt}%
        \global\@funsidebarfalse
        \global\@funfirstcolumnsopentrue
    \fi
}
\pretocmd{\@sect}{\@funstartfirstpagecolumns}{}{}

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

\definecolor{orcidlogocol}{HTML}{A6CE39}
\newcommand{\orcid}[1]{\,\href{https://orcid.org/#1}{\textsuperscript{\textcolor{orcidlogocol}{\faOrcid}}}}

\DeclareFontFamily{U}{stix2bb}{}
\DeclareFontShape{U}{stix2bb}{m}{n} {<-> stix2-mathbb}{}
\NewDocumentCommand{\indicator}{}{\text{\usefont{U}{stix2bb}{m}{n}1}}

% \RequirePackage[columnwise]{lineno}
% \AtBeginDocument{%
%   \setlength\linenumbersep{4pt}%
%   \linenumbers
% }
