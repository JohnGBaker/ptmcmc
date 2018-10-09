#Note: autoconf-archive must be installed

AC_INIT(ptmcmc, 1.0)

dnl Switch to a C++ compiler, and check if it works.
AC_ARG_ENABLE([openmp],
  [AS_HELP_STRING([--disable-openmp], [Enable OpenMP support @<:@check@:>@])],
  [:],
  [enable_openmp=check]
)
  
AC_LANG(C++)
AC_PROG_CXX

AX_CXX_COMPILE_STDCXX_11
AC_OPENMP
AS_ECHO([OPENMP_CXXFLAGS=${OPENMP_CXXFLAGS}])
AC_SUBST([OPENMP_CXXFLAGS],[${OPENMP_CXXFLAGS}])
AS_IF([test "$enable_openmp" != "no"],
	    AC_MSG_CHECKING([for OpenMP]);
	    AS_IF([ test -n "${OPENMP_CXXFLAGS}"],
	        AC_MSG_RESULT([yes]),
	    	AC_MSG_RESULT([no]);
		AC_MSG_ERROR([Specified compiler does not support OpenMP. Try setting CXX=better_compiler or run configure --disable-openmp])
		)
	,AC_MSG_NOTICE([OpenMP Disabled. But it is may be essential to build the code. ])
	)
dnl Process Makefile.in to create Makefile
AC_OUTPUT(Makefile.ac)
