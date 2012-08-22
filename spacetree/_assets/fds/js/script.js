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

        function log(msg) {
            var logger = $('#spacetree .logger');
            logger.val(function (_, value) {
                return $.trim(value + '\n' + msg);
            }).scrollTop(logger.prop('scrollHeight'));
        }

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
                type: 'rectangle', // NOTE: ellipse looks cool
                width: 150
            },
            Edge: {
                color: '#23a4ff',
                lineWidth: 2,
                overridable: true,
                type: 'bezier'
            },
            onBeforeCompute: function (node) {
                log("Loading '" + node.name + "'");
            },
            onAfterCompute: $.noop,
            onCreateLabel: function (label, node) {
                $(label).attr({
                    id: node.id
                }).click(function () {
                    ////if (normal.checked) {
                    ////    st.onClick(node.id);
                    ////} else {
                    st.setRoot(node.id, 'animate');
                    ////}
                }).css({
                    color: '#444',
                    cursor: 'pointer',
                    fontSize: '0.8em',
                    paddingTop: '13px',
                    textAlign: 'center'
                }).html(node.name).height(0).width(175);
            },
            onBeforePlotNode: function (node) {
                if (node.selected) {
                    ////node.data.$color = '#ff7';
                } else {
                    node.data.$color = node.data.color || '#aaa';
                    ////delete node.data.$color;
                    ////if (!node.anySubnode('exist')) {
                    ////    var count = 0;
                    ////    node.eachSubnode(function(n) { count++; });
                    ////    node.data.$color = ['#baa', '#caa', '#daa', '#eaa', '#faa'][count];
                    ////}
                }
            },
            onBeforePlotLine: function (adj) {
                if (adj.nodeFrom.selected && adj.nodeTo.selected) {
                    adj.data.$color = '#eed';
                    adj.data.$lineWidth = 3;
                } else {
                    delete adj.data.$color;
                    delete adj.data.$lineWidth;
                }
            }
        });

        // Add event handlers to switch spacetree orientation.
        var orient = $('#r-top,#r-bottom,#r-left,#r-right,#s-normal');
        orient.change(function () {
            if (!this.checked) return;
            orient.prop('disabled', true);
            st.switchPosition(this.value, 'animate', {
                onComplete: function () {
                    orient.removeProp('disabled');
                }
            });
        });

        $.getJSON('/_assets/ajax/tree.json', function (json) {
            st.loadJSON(json);
            st.toJSON('graph');
            st.compute();
            st.geom.translate(new $jit.Complex(-200, 0), 'current');
            //st.onClick(st.root);
            st.onClick('root');
        });

        $.getJSON('/_assets/ajax/node_list.json', function (node_list) {
            $('#nodes').autocomplete({
                source: node_list,
                select: function (event, ui) {
                    // TODO: Make onClick's work, so that on load of json it
                    //       clicks on 'root' node, and on autocomplete it
                    //       loads selected node.
                    st.onClick(ui.item.label);
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
