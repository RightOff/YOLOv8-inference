# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/clh/YOLOv8-inference

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/clh/YOLOv8-inference/build

# Utility rule file for yolov8_autogen.

# Include the progress variables for this target.
include CMakeFiles/yolov8_autogen.dir/progress.make

CMakeFiles/yolov8_autogen:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/clh/YOLOv8-inference/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Automatic MOC and UIC for target yolov8"
	/usr/bin/cmake -E cmake_autogen /home/clh/YOLOv8-inference/build/CMakeFiles/yolov8_autogen.dir/AutogenInfo.json Release

yolov8_autogen: CMakeFiles/yolov8_autogen
yolov8_autogen: CMakeFiles/yolov8_autogen.dir/build.make

.PHONY : yolov8_autogen

# Rule to build all files generated by this target.
CMakeFiles/yolov8_autogen.dir/build: yolov8_autogen

.PHONY : CMakeFiles/yolov8_autogen.dir/build

CMakeFiles/yolov8_autogen.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/yolov8_autogen.dir/cmake_clean.cmake
.PHONY : CMakeFiles/yolov8_autogen.dir/clean

CMakeFiles/yolov8_autogen.dir/depend:
	cd /home/clh/YOLOv8-inference/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/clh/YOLOv8-inference /home/clh/YOLOv8-inference /home/clh/YOLOv8-inference/build /home/clh/YOLOv8-inference/build /home/clh/YOLOv8-inference/build/CMakeFiles/yolov8_autogen.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/yolov8_autogen.dir/depend

