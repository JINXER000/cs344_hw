#ifndef COMPARE_H
#define COMPARE_H

void compareImages(std::string reference_filename, std::string test_filename, bool useEpsCheck,
				   double perPixelError, double globalError);

#endif
