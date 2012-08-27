// -----------------------------------------------------------------------------
// POLARIS Flight Data Parameter Tree
// Copyright © 2012, Flight Data Services Ltd.
// -----------------------------------------------------------------------------

/*jshint browser: true, es5: true, indent: 4, jquery: true, strict: true */
/*global $jit: true, tableToGrid: true */

(function ($, window, document, undefined) {
    'use strict';

// -----------------------------------------------------------------------------
// Spacetree
// -----------------------------------------------------------------------------

    $(function () {

        // Hack to determine what <canvas> support is available:
        var labelType, useGradients, nativeTextSupport, animate;
        (function () {
            var ua = navigator.userAgent,
                iStuff = ua.match(/iPhone/i) || ua.match(/iPad/i),
                typeOfCanvas = typeof HTMLCanvasElement,
                nativeCanvasSupport = (typeOfCanvas == 'object' || typeOfCanvas == 'function'),
                textSupport = nativeCanvasSupport && (typeof document.createElement('canvas').getContext('2d').fillText == 'function');
            //I'm setting this based on the fact that ExCanvas provides text support for IE
            //and that as of today iPhone/iPad current text support is lame
            labelType = (!nativeCanvasSupport || (textSupport && !iStuff)) ? 'Native' : 'HTML';
            nativeTextSupport = labelType == 'Native';
            useGradients = nativeCanvasSupport;
            animate = !(iStuff || !nativeCanvasSupport);
        })();

        // Initialise the spacetree:
        var st = new $jit.ST({
            constrained: true,
            duration: 500,
            injectInto: 'spacetree-canvas',
            levelDistance: 50,
            levelsToShow: 1,
            transition: $jit.Trans.Quart.easeInOut,
            Navigation: {
                enable: true,
                panning: true
            },
            Node: {
                autoHeight: true,
                autoWidth: false,
                color: '#aaa',
                height: 0,
                overridable: true,
                type: 'rectangle',
                width: 180
            },
            Edge: {
                color: '#23a4ff',
                lineWidth: 2,
                overridable: true,
                type: 'bezier'
            },
            onBeforeCompute: function (node) {
                var logger = $('#spacetree .logger'),
                    msg = "Loading '" + node.name + "'";
                logger.val(function (_, value) {
                    return $.trim(value + '\n' + msg);
                }).scrollTop(logger.prop('scrollHeight'));
            },
            onAfterCompute: $.noop,
            onCreateLabel: function (label, node) {
                $(label).attr({
                    id: node.id
                }).click(function () {
                    if ($('#s-normal').prop('checked')) {
                        st.onClick(node.id);
                    } else {
                        st.setRoot(node.id, 'animate');
                    }
                }).css({
                    boxShadow: '0 0 5px #ccc',
                    color: '#444',
                    cursor: 'pointer',
                    fontSize: '85%',
                    lineHeight: 1,
                    padding: '10px',
                    textAlign: 'center',
                    textShadow: '0 0 2px #ccc'
                }).html(node.name).height('auto').width(180);
            },
            onBeforePlotNode: function (node) {
                node.data.$color = node.data.color || '#aaa';
            },
            onBeforePlotLine: function (adj) {
                if (adj.nodeFrom.selected && adj.nodeTo.selected) {
                    adj.data.$color = '#23a4ff';
                    adj.data.$lineWidth = 2;
                } else {
                    delete adj.data.$color;
                    delete adj.data.$lineWidth;
                }
            }
        });

        // Add event handlers to switch spacetree orientation:
        var orient = $('#r-top,#r-bottom,#r-left,#r-right');
        orient.change(function () {
            if (!this.checked) return;
            orient.prop('disabled', true);
            st.switchPosition(this.value, 'animate', {
                onComplete: function () {
                    orient.removeProp('disabled');
                }
            });
        });

        // Fetch the data for the spacetree as JSON:
        $.getJSON('/_assets/ajax/tree.json', function (json) {
            st.loadJSON(json);
            st.toJSON('graph');
            st.compute();
            st.geom.translate(new $jit.Complex(-200, 0), 'current');
            // FIXME: Improve initial display as 'root' is messy:
            st.onClick('root');  // st.onClick(st.root);
        });

        // Fetch the data for the parameter search as JSON:
        $.getJSON('/_assets/ajax/node_list.json', function (node_list) {
            $('#spacetree-search').autocomplete({
                source: node_list,
                select: function (event, ui) {
                    st.setRoot(ui.item.label, 'replot');
                    $(this).val('').blur();
                    return false;
                }
            });
        });

    });

// -----------------------------------------------------------------------------
// Parameters
// -----------------------------------------------------------------------------

    $(function () {

        $('table.autogrid').each(function (index, element) {
            var table = $(element),
                columns = [],
                options = $.extend(true, {}, {
                    altRows: true,
                    autowidth: true,
                    emptyrecords: 'No records found.',
                    gridview: true, // Breaks afterInsertRow, subGrid, treeGrid.
                    ignoreCase: true,
                    loadtext: 'Loading…',
                    recordtext: 'Viewing {0} - {1} of {2}',
                    rowList: [20, 50, 100, 200, 500],
                    sortable: true,
                    viewrecords: true,
                    caption: table.attr('title') || '',
                    height: table.data('height') || 150,
                    pager: table.data('pager') || '',
                    rowNum: table.data('rowNum') || 20,
                    shrinkToFit: table.data('shrinkToFit') || false
                });
            // Loop over the table headers and detect column options.
            table.find('thead th').each(function (index, element) {
                var th = $(element),
                    // FIXME: Lowercased would be more sensible - issue with tableToGrid()?
                    id = $.trim(th.text()).replace(/ /g, '_'),
                    data = th.data(),
                    settings = {
                        name: id,
                        index: id
                    };
                // Add all data items as column settings.
                // Loop over data items and look for functions or JSON strings.
                $.each(data, function (k, v) {
                    // FIXME: Support functions...
                    //if ($.type(v) === 'string' && v.match(/^fn:/)) {
                    //    v = v.substr(3).findObject();
                    //}
                    settings[k] = v;
                });
                columns.push(settings);
            });
            // Add the column model to the grid options.
            if (columns) {
                $.extend(options, {
                    colModel: columns
                });
            }
            // Create the grid from the table.
            tableToGrid(table, options);
            // XXX: Workaround as tableToGrid causes all rows to show initially...
            if (table.data('pager')) {
                table.jqGrid('setGridParam', {
                    page: 1
                }).trigger('reloadGrid');
            }
            // Add the filter toolbar (if requested).
            if (table.data('filterToolbar')) {
                table.jqGrid('filterToolbar', {
                    defaultSearch: 'cn',
                    searchOnEnter: false,
                    stringResult: true
                });
            }
        });

    });

// -----------------------------------------------------------------------------

}(jQuery, window, document));

// -----------------------------------------------------------------------------
// vim:et:ft=javascript:nowrap:sts=4:sw=4:ts=4
