#cython: boundscheck=False
#cython: wraparound=False
#cython: profile=True

cimport numpy as np
cimport cython
import numpy as np
from itertools import izip

np.import_array()

class CSMat():
    """Compressed sparse (CSR/CSC) and coordinate list (COO) formats.

    Builds sparse matrix in COO format first (good for incremental
    construction) and then converts to CSR/CSC for efficient row/column access.

    Must specify number of rows and columns in advance.

    Object cannot "grow" in shape.

    enable_indices is ignored.
    """
    def __init__(self, np.uint32_t rows, np.uint32_t cols, dtype=float, 
            enable_indices=True):
        cdef np.ndarray[np.double_t, ndim=1, mode='c'] _values
        cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] _pkd_ax
        cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] _unpkd_ax
        
        self.shape = (rows, cols) 
        self.dtype = dtype # casting is minimal, trust the programmer...
        self._order = "coo"

        # coordinate list format
        # CAN CDEF THESE AS LISTS BUT THEN NOT ACCESSIBLE FROM PYTHON
        self._coo_values = []
        self._coo_rows = []
        self._coo_cols = []

        _values = np.empty(0, dtype=np.double)
        _pkd_ax = np.empty(0, dtype=np.uint32)
        _unpkd_ax = np.empty(0, dtype=np.uint32)
        self._values = _values
        self._pkd_ax = _pkd_ax
        self._unpkd_ax = _unpkd_ax
    
    def _get_size(self):
        """Returns the number of non-zero elements stored (NNZ)."""
        if self.hasUpdates():
            self.absorbUpdates()

        if self._order == "coo":
            return len(self._coo_values)
        else:
            return self._values.size
    size = property(_get_size)

    def transpose(self):
        """Transpose self"""
        cdef str rebuild
        cdef object new_self ### ONCE TYPEDEFED THIS CAN BE CHANGED FROM OBJECT TO CSMAT

        new_self = self.copy()

        if new_self._order != "coo":
            rebuild = new_self._order
            new_self.convert("coo")
        else:
            rebuild = None

        new_self.shape = (new_self.shape[1], new_self.shape[0])
        tmp = new_self._coo_rows
        new_self._coo_rows = new_self._coo_cols
        new_self._coo_cols = tmp

        if rebuild is not None:
            new_self.convert(rebuild)

        return new_self
    T = property(transpose)

    def update(self, dict data):
        """Update from a dict"""
        cdef np.uint32_t r,c
        cdef np.double_t v

        for (r,c),v in data.iteritems():
            self[(r,c)] = v

    def bulkCOOUpdate(self, list rows, list cols, list values):
        """Stages data in COO format. Expects 3 iterables aligned by index."""
        cdef np.uint32_t r, c, i
        cdef np.double_t v
        
        for i from 0 <= i < len(values):
            v = values[i]
            if v != 0:
                self._coo_values.append(v)
                self._coo_rows.append(rows[i])
                self._coo_cols.append(cols[i])

    def hasUpdates(self):
        """Returns true if it appears there are updates"""
        if len(self._coo_values) != 0:
            return True
        else:
            return False

    def absorbUpdates(self):
        """If there are COO values not in CS form, pack them in"""
        cdef str order

        if self._order == 'coo':
            return
        
        if not self._coo_values:
            return

        # possibly a better way to do this
        order = self._order
        self.convert("coo")
        self.convert(order)

    def convert(self, str to_order):
        """Converts to csc <-> csr, csc <-> coo, csr <-> coo"""
        if self._order == to_order:
            return

        if self._order == "coo":
            self._buildCSfromCOO(to_order)
        else:
            if to_order == "coo":
                self._buildCOOfromCS()
            else:
                self._buildCSfromCS()

    def getRow(self, np.uint32_t row):
        """Returns a row in Sparse COO form"""
        cdef np.uint32_t n_rows, n_cols, start, stop, n_vals, i

        if row >= self.shape[0] or row < 0:
            raise IndexError, "Row %d is out of bounds!" % row

        if self.hasUpdates():
            self.absorbUpdates()

        n_rows,n_cols = self.shape
        v = self.__class__(1, n_cols, dtype=self.dtype)

        if self._order != "csr":
            self.convert("csr")

        start = self._pkd_ax[row]
        stop = self._pkd_ax[row + 1]
        n_vals = stop - start
    
        # direct to CSR
        v._order = "csr"
        v._pkd_ax = np.empty(2, dtype=np.uint32)
        v._pkd_ax[0] = 0
        v._pkd_ax[1] = n_vals
      
        v._unpkd_ax = np.empty(n_vals, dtype=np.uint32)
        v._values = np.empty(n_vals, dtype=np.double)
        for i from 0 <= i < n_vals:
            v._unpkd_ax[i] = self._unpkd_ax[i + start]
            v._values[i] = self._values[i + start]
        
        return v

    def getCol(self, np.uint32_t col):
        """Return a col in CSMat form"""
        cdef np.uint32_t n_rows, n_cols, start, stop, n_vals, i

        if col >= self.shape[1] or col < 0:
            raise IndexError, "Col %d is out of bounds!" % col

        if self.hasUpdates():
            self.absorbUpdates()

        n_rows,n_cols = self.shape
        v = self.__class__(n_rows, 1, dtype=self.dtype)

        if self._order != "csc":
            self.convert("csc")

        start = self._pkd_ax[col]
        stop = self._pkd_ax[col + 1]
        n_vals = stop - start

        # direct to CSC
        v._order = "csc"
        v._pkd_ax = np.empty(2, dtype=np.uint32)
        v._pkd_ax[0] = 0
        v._pkd_ax[1] = n_vals
      
        v._unpkd_ax = np.empty(n_vals, dtype=np.uint32)
        v._values = np.empty(n_vals, dtype=np.double)
        for i from 0 <= i < n_vals:
            v._unpkd_ax[i] = self._unpkd_ax[i + start]
            v._values[i] = self._values[i + start]
        
        return v

    def items(self):
        """returns [((r,c),v)]"""
        if self.hasUpdates():
            self.absorbUpdates()
        
        res = []
        if self._order == 'csr':
            return _items_csr(self._unpkd_ax, self._pkd_ax, self._values)
        elif self._order == 'csc':
            return _items_csc(self._unpkd_ax, self._pkd_ax, self._values)
        else:
            for r,c,v in izip(self._coo_rows, self._coo_cols, self._coo_values):
                res.append(((r,c),v))
        return res
    
    def iteritems(self):
        """Generator returning ((r,c),v)"""
        cdef np.uint32_t i
        cdef np.ndarray expanded

        if self.hasUpdates():
            self.absorbUpdates()
        
        if self._order == 'csr':
            # cannot easily cdef cythonize due to yield
            expanded = self._expand_compressed(self._pkd_ax)
            for i from 0 <= i < expanded.size:
                yield ((expanded[i], self._unpkd_ax[i]), self._values[i])
        elif self._order == 'csc':
            # cannot easily cdef cythonize due to yield
            expanded = self._expand_compressed(self._pkd_ax)
            for i from 0 <= i < expanded.size:
                yield ((self._unpkd_ax[i], expanded[i]), self._values[i])
        else:
            for i from 0 <= i < len(self._coo_values):
                yield ((self._coo_rows[i],self._coo_cols[i]),
                       self._coo_values[i])

    def __contains__(self, tuple args):
        """Return True if args are in self, false otherwise"""
        if self._getitem(args) == (None, None, None):
            return False
        else:
            return True

    def copy(self):
        """Return a copy of self"""
        new_self = self.__class__(*self.shape, dtype=self.dtype)
        new_self._coo_rows = self._coo_rows[:]
        new_self._coo_cols = self._coo_cols[:]
        new_self._coo_values = self._coo_values[:]
        new_self._pkd_ax = self._pkd_ax.copy()
        new_self._unpkd_ax = self._unpkd_ax.copy()
        new_self._values = self._values.copy()
        new_self._order = self._order[:]
        return new_self

    def __eq__(self, object other):
        """Returns true if both CSMats are the same"""
        if self.shape != other.shape:
            return False

        if self.hasUpdates():
            self.absorbUpdates()
        if other.hasUpdates():
            other.absorbUpdates()

        if self.shape[1] == 1:
            self.convert("csc")
            other.convert("csc")
        else:
            if self._order != "csr":
                self.convert("csr")
            if other._order != "csr":
                other.convert("csr")
    
        if _eq(self, other) == 1:
            return True
        else:
            return False

    def __ne__(self, object other):
        """Return true if both CSMats are not equal"""
        return not (self == other)

    def __str__(self):
        """dump priv data"""
        l = []
        l.append(self._order)
        l.append("_coo_values\t" + '\t'.join(map(str, self._coo_values)))
        l.append("_coo_rows\t" + '\t'.join(map(str, self._coo_rows)))
        l.append("_coo_cols\t" + '\t'.join(map(str, self._coo_cols)))
        l.append("_values\t" + '\t'.join(map(str, self._values)))
        l.append("_pkd_ax\t" + '\t'.join(map(str, self._pkd_ax)))
        l.append("_unpkd_ax\t" + '\t'.join(map(str, self._unpkd_ax)))
        return '\n'.join(l)

    def __setitem__(self, tuple args, np.double_t value):
        """Wrap setitem, complain if out of bounds"""
        try:
            row,col = args
        except:
            # fast support foo[5] = 10, like numpy 1d vectors
            col = args
            row = 0
            args = (row,col)

        if row >= self.shape[0]:
            raise IndexError, "Row %d is out of bounds!" % row
        if col >= self.shape[1]:
            raise IndexError, "Col %d is out of bounds!" % col

        if value == 0:
            if args in self:
                raise ValueError("Cannot set an existing non-zero element to "
                                 "zero.")
        else:
            res = self._getitem(args)
            if res == (None, None, None):
                self._coo_rows.append(row)
                self._coo_cols.append(col)
                self._coo_values.append(value)
            else:
                if self._order == "coo":
                    self._coo_values[res[0]] = value
                else:
                    self._values[res[2]] = value

    def __getitem__(self, tuple args):
        """Wrap getitem to handle slices"""
        try:
            row,col = args
        except TypeError:
            raise IndexError, "Must specify (row, col)"

        if isinstance(row, slice): 
            if row.start is None and row.stop is None:
                return self.getCol(col)
            else:
                raise AttributeError, "Can only handle full : slices per axis"
        elif isinstance(col, slice):
            if col.start is None and col.stop is None:
                return self.getRow(row)
            else:
                raise AttributeError, "Can only handle full : slices per axis"
        else:
            if row >= self.shape[0] or row < 0:
                raise IndexError, "Row out of bounds!"
            if col >= self.shape[1] or col < 0:
                raise IndexError, "Col out of bounds!"

            res = self._getitem(args)
            if res == (None,None,None):
                return self.dtype(0)
            else:
                if self._order == 'coo':
                    return self._coo_values[res[0]]
                else:
                    return self._values[res[2]] 
                
    def _getitem(self, tuple args):
        """Mine for an item
        
        if order is csc | csr, returns
        pkd_ax_idx, unpkd_ax_idx, values_idx 

        if order is coo, returns
        rows_idx, cols_idx, values_idx (all the same thing...)
        """
        cdef str order 
        cdef tuple result
        cdef np.uint32_t row, col

        if self.hasUpdates():
            self.absorbUpdates()

        row, col = args
        if self._order == "coo":
            result = _getitem_coo(self._coo_rows, self._coo_cols, row, col)
        else:
            order = self._order
            result = _getitem_cs(self._pkd_ax, self._unpkd_ax, row, col, order)
        
        if result[0] == -1:
            return (None, None, None)
        else:
            return result

    def _buildCSfromCS(self):
        """Convert csc <-> csr"""
        cdef tuple cs
        cdef np.ndarray expanded

        expanded = self._expand_compressed(self._pkd_ax)
        if self._order == "csr": # to CSC
            cs = _toCS_ndarray(self.shape[1], self._unpkd_ax, expanded, 
                                self._values)
            self._pkd_ax, self._unpkd_ax, self._values = cs
            self._order = "csc"

        elif self._order == "csc": # to CSR
            cs = _toCS_ndarray(self.shape[0], self._unpkd_ax, expanded,
                                self._values)
            self._pkd_ax, self._unpkd_ax, self._values = cs
            self._order = "csr"
            
    def _buildCOOfromCS(self):
        """Constructs a COO representation from CSC or CSR
        
        Invalidates existing CSC or CSR representation
        """
        cdef tuple coo
        cdef list coo_rows, coo_cols, coo_values

        coo = self._toCOO(self._pkd_ax,self._unpkd_ax,self._values,self._order)
        coo_rows, coo_cols, coo_values = coo
        self._coo_rows.extend(coo_rows)
        self._coo_cols.extend(coo_cols)
        self._coo_values.extend(coo_values)

        self._values = np.array([], dtype=self.dtype)
        self._pkd_ax = np.array([], dtype=np.uint32)
        self._unpkd_ax = np.array([], dtype=np.uint32)
        
        self._order = "coo"

    def _toCOO(self, np.ndarray[np.uint32_t, ndim=1, mode='c'] pkd_ax, 
                     np.ndarray[np.uint32_t, ndim=1, mode='c'] unpkd_ax, 
                     np.ndarray[np.double_t, ndim=1, mode='c'] values, 
                     str current_order):
        """Returns rows, cols, values"""
        cdef list coo_values, coo_cols, coo_rows, expanded_ax

        coo_values = list(values)
        expanded_ax = list(self._expand_compressed(pkd_ax))
        
        if current_order == 'csr':
            coo_cols = list(unpkd_ax)
            coo_rows = expanded_ax

        elif current_order == 'csc':
            coo_rows = list(unpkd_ax)
            coo_cols = expanded_ax
        else:
            raise ValueError, "Unknown order: %s" % current_order

        return (coo_rows, coo_cols, coo_values)
        
    def _buildCSfromCOO(self, str order):
        """Build a sparse representation

        order is either csc or csr

        Returns instantly if is stable, throws ValueError if the sparse rep
        is already built
        """
        cdef tuple cs

        if order == 'csr':
            cs = _toCS_pylist(self.shape[0], self._coo_rows, 
                               self._coo_cols, self._coo_values)
            self._pkd_ax, self._unpkd_ax, self._values = cs
        elif order == 'csc':
            cs = _toCS_pylist(self.shape[1], self._coo_cols, 
                               self._coo_rows, self._coo_values)
            self._pkd_ax, self._unpkd_ax, self._values = cs 
        else:
            raise ValueError, "Unknown order: %s" % order

        self._coo_rows = []
        self._coo_cols = []
        self._coo_values = []
        self._order = order

    def _expand_compressed(self, np.ndarray[np.uint32_t, ndim=1, mode='c'] pkd_ax):
        return _expand_compressed(pkd_ax)

cdef tuple _toCS_ndarray(np.uint32_t axis_len, 
                         np.ndarray[np.uint32_t, ndim=1, mode='c'] to_pack_in, 
                         np.ndarray[np.uint32_t, ndim=1, mode='c'] not_pack_in, 
                         np.ndarray[np.double_t, ndim=1, mode='c'] values_in):
    """Returns packed_axis, unpacked_axis, values"""
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] values
    cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] unpkd_ax, tmp_pkd, tmp_pkd_sorted
    cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] pkd_ax
    cdef np.ndarray[np.long_t, ndim=1, mode='c'] order
    cdef np.uint32_t nz, i, idx

    # number of nonzero values
    nz = values_in.size
    
    # initialize
    values = np.empty(nz, dtype=np.double)
    unpkd_ax = np.empty(nz, dtype=np.uint32)
    tmp_pkd = np.empty(nz, dtype=np.uint32)
    tmp_pkd_sorted = np.empty(nz, dtype=np.uint32)
    
    if nz == 0:
        return (np.zeros(axis_len + 1, dtype=np.uint32), unpkd_ax, values)
    
    # loadup tmp rows into numpy to drive sorting
    for i from 0 <= i < nz:
        tmp_pkd[i] = np.uint32(to_pack_in[i])

    # get sorted order
    order = np.argsort(tmp_pkd)

    # populate, avoid multiple takes
    for i from 0 <= i < nz:
        idx = order[i]
        tmp_pkd_sorted[i] = tmp_pkd[idx]
        unpkd_ax[i] = not_pack_in[idx]
        values[i] = values_in[idx]
    
    pkd_ax = _cs_pack(nz, axis_len, tmp_pkd_sorted)
    return (pkd_ax, unpkd_ax, values)

cdef tuple _toCS_pylist(np.uint32_t axis_len, list to_pack_in, list not_pack_in, 
                      list values_in):
    """Returns packed_axis, unpacked_axis, values"""
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] values 
    cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] unpkd_ax, tmp_pkd, tmp_pkd_sorted 
    cdef np.ndarray[np.long_t, ndim=1, mode='c'] order
    cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] pkd_ax
    cdef np.uint32_t nz, i, idx

    # number of nonzero values
    nz = len(values_in)

    # initialize
    values = np.empty(nz, dtype=np.float64)
    unpkd_ax = np.empty(nz, dtype=np.uint32)
    tmp_pkd = np.empty(nz, dtype=np.uint32)
    tmp_pkd_sorted = np.empty(nz, dtype=np.uint32)

    if nz == 0:
        return (np.zeros(axis_len + 1, dtype=np.uint32), unpkd_ax, values)
    
    # loadup tmp pkd into numpy to drive sorting
    for i from 0 <= i < nz:
        tmp_pkd[i] = np.uint32(to_pack_in[i])

    # get sorted order
    order = np.argsort(tmp_pkd)

    # populate
    for i from 0 <= i < nz:
        idx = order[i]
        values[i] = values_in[idx] 
        tmp_pkd_sorted[i] = tmp_pkd[idx]
        unpkd_ax[i] = not_pack_in[idx]
    
    pkd_ax = _cs_pack(nz, axis_len, tmp_pkd_sorted)
    return (pkd_ax, unpkd_ax, values)

cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] _cs_pack(np.uint32_t nz, 
                        np.uint32_t axis_len, 
                        np.ndarray[np.uint32_t, ndim=1, mode='c'] ax_sorted):
    """Pack an axis"""
    cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] pkd_ax
    cdef np.uint32_t pos = 0
    cdef np.int32_t v_last = -1
    cdef np.uint32_t expected_axis_idx = 0
    cdef np.uint32_t i, j, v, num_empty, num_trailing

    #### these calls are generating a LOT more C code than expected. wtf?
    #### should verify function decorators and how np methods are used
    pkd_ax = np.empty(axis_len + 1, dtype=np.uint32)
    for i from 0 <= i < nz:
        v = ax_sorted[i]
    
        if v == v_last:
            continue
        else:
            v_last = v
        pkd_ax[v] = i

        num_empty = v - expected_axis_idx
        if num_empty > 0:
            for j from 1 <= j <= num_empty:
                pkd_ax[v - j] = i
        expected_axis_idx = v + 1

    pkd_ax[v+1] = i   
    
    # get dangling zero'd columns if they exist
    num_empty = axis_len - expected_axis_idx
    if num_empty > 0:
        for j from 1 <= j <= num_empty:
            pkd_ax[v+1 + j] = i
    return pkd_ax

cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] _expand_compressed(np.ndarray[np.uint32_t, ndim=1] pkd_ax):
    """Expands packed axis"""
    cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] expanded
    cdef np.uint32_t pos
    cdef np.uint32_t last_idx 
    cdef np.uint32_t idx, start, end
    cdef np.uint32_t pos_at_idx

    end = pkd_ax.size - 1
    start = 1

    expanded = np.empty(pkd_ax[end], dtype=np.uint32)
    last_idx = 0
    pos = 0
    
    for idx from start <= idx <= end:
        pos_at_idx = pkd_ax[idx]
        expanded[last_idx:pos_at_idx] = pos
        pos += 1
        last_idx = pos_at_idx
    return expanded

cdef list _items_csr(np.ndarray[np.uint32_t, ndim=1, mode='c'] unpkd_ax,
                     np.ndarray[np.uint32_t, ndim=1, mode='c'] pkd_ax,
                     np.ndarray[np.double_t, ndim=1, mode='c'] values):
    cdef np.uint32_t i
    cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] expanded
    cdef list res = []
   
    expanded = _expand_compressed(pkd_ax)

    # it would not fit the current api, but could allocate a matrix upfront
    # thats 3xNZ... would be faster and more mem efficient

    for i from 0 <= i < expanded.size:
        res.append(((expanded[i], unpkd_ax[i]), values[i]))
    
    return res

cdef list _items_csc(np.ndarray[np.uint32_t, ndim=1, mode='c'] unpkd_ax,
                     np.ndarray[np.uint32_t, ndim=1, mode='c'] pkd_ax,
                     np.ndarray[np.double_t, ndim=1, mode='c'] values):
    cdef np.uint32_t i
    cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] expanded
    cdef list res = [] 
    
    expanded = _expand_compressed(pkd_ax)

    # it would not fit the current api, but could allocate a matrix upfront
    # thats 3xNZ... would be faster and more mem efficient

    for i from 0 <= i < expanded.size:
        res.append(((unpkd_ax[i], expanded[i]), values[i]))
    return res

cdef unsigned int _eq(object self, object other):
    """test equality between two CSMat objects"""
    cdef np.ndarray[Py_ssize_t, ndim=1, mode='c'] self_order, other_order
    cdef np.uint32_t i, svi, ovi
    cdef np.double_t svd, ovd
    cdef unsigned int self_size, other_size

    # pulling the value out first results in a better cythonizing by allowing
    # cython to use a C test in the if statement instead of a general purpose
    # python one
    self_size = self._pkd_ax.size
    other_size = other._pkd_ax.size
    if self_size != other_size:
        return False

    self_size = self._unpkd_ax.size
    other_size = other._unpkd_ax.size
    if self_size != other_size:
        return False

    # can do these three checks in a single loop if necessary
    if (self._pkd_ax != other._pkd_ax).any():
        return False

    self_order = np.argsort(self._unpkd_ax)
    other_order = np.argsort(other._unpkd_ax)
    for i from 0 <= i < self._unpkd_ax.size:
        svi = self._unpkd_ax[self_order[i]]
        ovi = other._unpkd_ax[other_order[i]]
            
        svd = self._values[self_order[i]]
        ovd = other._values[other_order[i]]

        if svi != ovi:
            return False
        if svd != ovd:
            return False

    return True

cdef tuple _getitem_cs(np.ndarray[np.uint32_t, ndim=1, mode='c'] pkd_ax, 
                       np.ndarray[np.uint32_t, ndim=1, mode='c'] unpkd_ax,
                       np.uint32_t row, np.uint32_t col, str order):
    """get an item from a compressed structure or (-1,-1,-1) if not found"""
    cdef np.uint32_t start, stop, i, c
    
    if order == 'csr':
        start = pkd_ax[row]
        stop = pkd_ax[row+1]

        for i from start <= i < stop:
            if unpkd_ax[i] == col:
                return (row, i, i)
    elif order == 'csc':
        start = pkd_ax[col]
        stop = pkd_ax[col+1]
        
        for i from start <= i < stop:
            if unpkd_ax[i] == row:
                return (i, col, i)
    return (-1, -1, -1)

cdef tuple _getitem_coo(list rows, list cols, np.uint32_t row, np.uint32_t col):
    """get an item from a COO, or return (-1, -1, -1) if not found"""
    cdef np.uint32_t i, r, c, nz
    
    nz = len(rows)

    for i from 0 <= i < nz:
        r = rows[i]
        c = cols[i]
        if r == row and c == col:
            return (i, i, i)
    return (-1, -1, -1)

def nparray_to_csmat(data, dtype=float):
    """Convert a numpy array to a CSMat"""
    if len(data.shape) == 1:
        return nparray_to_csmat_1d(data)
    else:
        return nparray_to_csmat_2d(data)

cdef object nparray_to_csmat_1d(np.ndarray[np.double_t,ndim=1,mode='c'] data):
    """convert a 1d nparray to csmat"""
    cdef object mat = CSMat(1, data.shape[0])
    cdef np.uint32_t i
    cdef np.double_t v
    cdef list rows, cols, vals
    
    rows = []
    cols = []
    vals = []
    for i from 0 <= i < data.shape[0]:
        v = data[i]
        if v != 0:
            rows.append(0)
            cols.append(i)
            vals.append(v)
    mat._coo_rows = rows
    mat._coo_cols = cols
    mat._coo_values = vals
    return mat

cdef object nparray_to_csmat_2d(np.ndarray[np.double_t,ndim=2,mode='c'] data):
    """convert a 2d nparray to csmat"""
    cdef object mat = CSMat(data.shape[0], data.shape[1])
    cdef np.uint32_t i,j
    cdef np.double_t v
    cdef list rows, cols, vals

    rows = []
    cols = []
    vals = []

    # use wheres and takes?
    for i from 0 <= i < data.shape[0]:
        for j from 0 <= j < data.shape[1]:
            v = data[i,j]
            if v != 0:
                rows.append(i)
                cols.append(j)
                vals.append(v)
    mat._coo_rows = rows
    mat._coo_cols = cols
    mat._coo_values = vals
    return mat

def list_csmat_to_csmat(list data, dtype=float):
    """Takes a list of CSMats and creates a CSMat"""
    cdef int iscol
    cdef np.uint32_t n_cols, n_rows
    
    if data[0].shape[0] > data[0].shape[1]:
        is_col = True
        n_cols = len(data)
        n_rows = data[0].shape[0]
    else:
        is_col = False
        n_rows = len(data)
        n_cols = data[0].shape[1]

    if is_col == True:
        return _cy_list_csmat_to_csmat_iscol(data, n_rows, n_cols)
    else:
        return _cy_list_csmat_to_csmat_isrow(data, n_rows, n_cols)

cdef object _cy_list_csmat_to_csmat_iscol(list data, np.uint32_t n_rows, 
                                          np.uint32_t n_cols):
    """list csmat to csmat if currently transposed"""
    cdef object mat = CSMat(n_rows, n_cols)
    cdef list rows, cols, vals
    cdef np.uint32_t i
    cdef object row
    cdef tuple rcv

    rows = []
    cols = []
    vals = []

    for i from 0 <= i < n_rows:
        row = data[i]
        for rcv in row.items():
            rows.append(rcv[0][0])
            cols.append(i)
            vals.append(rcv[1])
    mat._coo_rows = rows
    mat._coo_cols = cols
    mat._coo_values = vals

    return mat

cdef object _cy_list_csmat_to_csmat_isrow(list data, np.uint32_t n_rows, 
                                          np.uint32_t n_cols):
    """list csmat to csmat"""
    cdef object mat = CSMat(n_rows, n_cols)
    cdef list rows, cols, vals
    cdef np.uint32_t i
    cdef object row
    cdef tuple rcv

    rows = []
    cols = []
    vals = []

    for i from 0 <= i < n_rows:
        row = data[i]
        for rcv in row.items():
            rows.append(i)
            cols.append(rcv[0][1])
            vals.append(rcv[1])
    mat._coo_rows = rows
    mat._coo_cols = cols
    mat._coo_values = vals

    return mat
