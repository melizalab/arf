---
title: arf specification
redirect_from:
  - /specifications/arf/
  - /spec:1/
---

The Advanced Recording Format (ARF) is a specification for storing
electrophysiological, acoustic, and behavioral data along with associated
metadata and derived quantities in a hierarchical structure. ARF is based on the
[HDF5 format](http://www.hdfgroup.org/HDF5/), allowing files to be accessed on a
wide range of operating systems and architectures.

-   Name: <https://meliza.org/spec:1/arf> (1/arf)
-   Editor: Dan Meliza (dan at meliza.org)
-   Version: 2.1
-   State:  released
-   MIME type: `application/vnd.meliza-org.arf; version=2.1`

## Licence

Copyright (c) 2010-2018 C Daniel Meliza.

This Specification is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3 of the License, or (at your option) any
later version.

This Specification is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details.

You should have received a copy of the GNU General Public License along with
this program; if not, see <http://www.gnu.org/licenses>.

## Change Process

This Specification is a free and open Consensus-Oriented Specification System ([COSS](https://rfc.unprotocols.org/spec:2/COSS/)).

Version numbering from 2.0 on must be [semantic](http://semver.org). Changes that
maintain backwards compatibility (i.e., that do not change or remove any
required fields and attributes) result in increments to the minor version.
Changes that break backwards compatibility must receive new major version
numbers. Changes that significantly alter the goals of the specification must
result in a new identifier, preferably with a reference to the version of this
document at the time of the fork.

## Language

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD",
"SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be
interpreted as described in [RFC 2119](http://tools.ietf.org/html/rfc2119).

## Goals and conceptual framework

The goal of ARF is to provide an open, unified, flexible, and portable format
for storing time-varying data, along with sufficient metadata to reconstruct
the conditions of how the data were recorded for many decades in the future.

Time-varying data can be represented in two ways
([Brillinger 2008](http://dx.doi.org/10.2307/3315583)):

-   **time series:** A quantitative physical property of a system (e.g., sound
    pressure or voltage) measured as a function of time. In digital
    computers, time series data are always sampled at discrete
    moments in time, usually at fixed intervals. The *sampling
    rate* of the data is the number of times per second the value
    is sampled.
-   **point process:** A sequence of events taking place at discrete times. In a
    simple point process, events are defined only by the times
    of occurrence. In a *marked* point process, additional
    information is associated with the events.

Bioacoustic, electrophysiological, and behavioral data can all be represented
in this framework. Some examples:

-   an acoustic recording is a time series of the sound pressure level detected
    by a microphone
-   an extracellular neural recording is a time series of the voltage measured by
    an electrode
-   a spike train is an unmarked point process defined by the times of the action
    potentials. A spike train could also be marked by the waveforms of the spikes.
-   stimuli presentations are a marked point process, with times indicating the
    onsets and offsets and marks indicating the identity of the presented stimulus
-   a series of behavioral events can be represented by a point process,
    optionally marked by measurable features of the event (location, intensity,
    etc)

Clearly all these types of data can be represented in computers as arrays. The
challenge is to organize and annotate the data in such a way that it can

1.  be unambiguously identified,
2.  be aligned with data from different sources,
3.  support a broad range of experimental designs, and
4.  be accessed with generic and widely available software

A major design goal of **ARF** is to be minimal. Only the attributes and
structural specifications needed to ensure that any type of data can be stored
are included. The rest is up to the user. This goal sets **ARF** apart from many
similar projects that attempt to explicitly deal with many use cases or that are
specialized for one or a few applications.

## Implementation

ARF files shall be in the HDF5 format, version 1.8 or later. HDF5 is critical to
providing flexibility and portability. It is available on multiple platforms and
supports automatic conversion of data types, allowing transparent access of data
across many architectures. HDF5 files support hierarchical organization of
datasets and metadata attributes. ARF specifies the layout used to store data
within this framework, while allowing the user to add metadata specific to an
application.

### Entries

An *entry* is defined as an abstract grouping of zero or more *datasets* that
all share a common start time. Each *entry* shall be represented by an HDF5
group. The group shall contain all the data objects associated with that entry,
stored as HDF5 datasets, and all the metadata associated with the entry, stored
as HDF5 attributes. The following attributes are required:

-   **timestamp:** The start time of the entry. This attribute shall consist of a
    two-element array with the first element indicating the number of
    seconds since January 1, 1970 UTC, and the second element
    indicating the rest of the elapsed time, in microseconds. Must
    have at least 64-bit integer precision.
-   **uuid:** A universally unique ID for the entry (see [RFC 4122](http://tools.ietf.org/html/rfc4122.html)). Must be stored
    as a 128-bit integer or a 36-byte `H5T_STRING` with `CTYPE` of
    `H5T_C_S1`. The latter is preferred as 128-bit integers are not
    supported on many platforms.

In addition, the following optional attributes are defined. They do not need to
be present in the group if not applicable, but if they are present they must
have a datatype with class `H5T_STRING` and `CTYPE` of `H5T_C_S1`. Encoding
must be ASCII or UTF-8 and match the value of `CSET`.

-   **animal:** Indicates the name or ID of the animal.
-   **experimenter:** Indicates the name or ID of the experimenter.
-   **protocol:** Comment field indicating the treatment, stimulus, or any other
    user-specified data.
-   **recuri:** The URI of an external database where `uuid` can be looked up.

### Datasets

A *dataset* is defined as a concrete time series or point process.  Multiple
datasets may be stored in an entry, and may be unequal in length or have
different *timebases*.

A *timebase* is defined by two quantities (with units), one of which is optional
under some circumstances. The required quantity is the *offset* of the data.
All time values in a dataset are relative to this time.  The default offset of
a dataset is the timestamp of the entry.  Individual datasets may have their
own offsets, which are calculated relative to the entry timestamp.

The second quantity in a timebase is the *sampling rate*, which allows discrete
times to be converted to real times. It is required if the data are sampled (as
in a time series) or if time values in a point process are in units of samples.
Only point proceses with real-valued units of time may omit the sampling rate.

Real-valued times must be in units of seconds. Discrete-valued times must be in
units of samples.

Each channel of data in an entry shall be represented by a separate HDF5
dataset. The format of each dataset depends on the type of data it stores.

#### Sampled data

Sampled data shall be stored as an N-dimensional array of scalar values
corresponding to the measurement at each sampling interval. The first dimension
of the array must correspond to time. The significance of additional dimensions
is unspecified. The `sampling_rate` attribute is required.

#### Event data

Event data may be stored in one of two formats. Simple event data should be
stored in a 1D array, with each element in the array indicating the time of the
event **relative to the start of the dataset**. Event datasets can be
distinguished from 1D sampled datasets because the `units` attribute must be
"samples" or "s".

Complex event data must be stored as arrays with a compound datatype (i.e., with
multiple fields). Only one field is required, `start`, which indicates the time
of the event and can be any numerical type.

Spike waveforms and features extracted from raw data should be stored in complex
event datasets, with the `start` field indicating the time of the spike and
additional array or scalar fields storing the waveforms and features.

A special case of event data are intervals, which are defined by a start and
stop time. In previous versions of the specification, intervals were considered
a separate data type, with two additional required fields, `name` (a string) and
`stop` (a time). This format is permitted in version 2.0, but intervals may also
be stored as separate start and stop events.

#### Dataset attributes

All datasets must have the following attributes.

- **units:** A string giving the units of the channel data, which should be in
  SI notation. May be an empty string for sampled data if units are not known.
  Event data must have units of "samples" (for a discrete timebase) or "s" (for
  a continuous timebase); sampled data must not use these units. For complex
  event data, this attribute must be an array, with each element of the array
  indicating the units of the associated field in the data.
- **datatype:** Indicates the source of data in the entry. Must have at least
  unsigned integer precision great enough to include all the values defined in
  5.2.4.

The following attribute is only required for datasets with a discrete timebase:

- **sampling\_rate:** A nonzero number indicating the sampling rate of the data,
  in samples per second (Hz). Required for all datasets with a sampled timebase.
  May be any numerical datatype.

The following attributes are optional:

- **offset:** Indicates the start time of the dataset relative to the start of
  the entry, defined by the timebase of the dataset. For discrete timebases, the
  units must be in samples; for continuous timebases, the units must be the same
  as the units of the dataset. If this attribute is missing, the offset shall be
  assumed to be zero.
- **uuid:** A universally unique ID for the dataset (see
  [RFC 4122](http://tools.ietf.org/html/rfc4122.html)). Multiple datasets in
  different entries of the same file may have the same uuid, indicating that
  they were obtained from the same source and experimental conditions. Must be
  stored as a 128-bit integer or a 36-byte `H5T_STRING` with `CTYPE` of
  `H5T_C_S1`. The latter is preferred as 128-bit integers are not supported on
  many platforms.

####  Datatypes

The `datatype` attribute is an integer code indicating the type of data in a
channel. This field is purely advisory: it specifies how the data should be
interpreted but does not imply any contract as to the dataspace or storage type
of the dataset. The following values are defined:


       0    UNDEFINED   undefined or unknown
       1    ACOUSTIC    acoustic
       2    EXTRAC_HP   extracellular, high-pass (single-unit or multi-unit)
       3    EXTRAC_LF   extracellular, local-field
       4    EXTRAC_EEG  extracellular, EEG
       5    INTRAC_CC   intracellular, current-clamp
       6    INTRAC_VC   intracellular, voltage-clamp
      23    EXTRAC_RAW  extracellular, wide-band
    1000    EVENT       generic event times
    1001    SPIKET      spike event times
    1002    BEHAVET     behavioral event times
    2000    INTERVAL    generic intervals
    2001    STIMI       stimulus presentation intervals
    2002    COMPONENTL  component (e.g. motif) labels

Values below 1000 are reserved for sampled data types.

### General structural rules

#### Top-level datasets

ARF files may have datasets in the root group. These must not associated with
any entry, but may be used to store structured data or metadata for the entire
file. For example, data recording software may keep a log of events. There are
no requirements for the datatype, dataspace, or attributes of these datasets.

#### Multiple linkages

Datasets must not be linked to more than one entry, as this would make the time
of the data undefined. Entries must not be multiply linked to the root HDF5
group. Entries may contain other entries, but their contents are not considered
part of the ARF data hierarchy.

### Extensions to the format

The above specification is a required minimum for a file to be in ARF format.
Additional attributes, groups, and datasets may be added, but must not conflict
with any attributes specified above. Because optional attributes may be forwards
incompatible with later versions due to namespace collision, their names should
be prefixed with the name of the application (e.g. 'jill\_sample\_count').

## Changes from previous versions

### version 2.1

An optional "uuid" attribute was added to the dataset specification. This
allows channels to be unambiguously identified as data sources for subsequent
analysis steps.

### version 2.0

The required "recid" attribute was dropped because it was unsuitable for an open
standard, and because it depended on an external database for uniqueness.
Instead, a "uuid" attribute was required.

Event data was defined to include both "simple" and "complex" events. Interval
data became a special case of complex event data. This was to allow data
collection programs to store more information about events, without forcing them
to use the strictly defined data type for intervals. The definition of a
distinct interval data type was dropped unceremoniously. Software reading the
INTERVAL, STIMI, and COMPONENTL should check for the existence of a 'stop'
field.

The times for event data were no longer required to be in units of seconds, and
the format was not required to be double-precision floating point. The
sampling\_rate attribute was required for event datasets where the units are in
samples.

Root-level datasets were explicitly allowed.

Semantic versioning was introduced.

To upgrade a file from version 1.1, add a uuid attribute to all entries, and a
sampling\_rate attribute to all event datasets that have units of samples.

### version 1.1

Catalogs were removed at the top level and in entries. The objects themselves
now carry all the metadata once in the catalog as attributes.

Multichannel datasets were deprecated in favor of multiple single-channel
datasets. Channels should only be grouped into single datasets when the data are
really inseparable (e.g. left and right channels). This greatly improved read
performance, at some expense in file size.

Entry groups were deprecated; datasets that start at different times but need to
be grouped together can be given an offset value indicating the interval between
the entry start time and the start of the data.

The attributes required by pytables were deprecated. Some interfaces may
continue to store them, but they were no longer required.
